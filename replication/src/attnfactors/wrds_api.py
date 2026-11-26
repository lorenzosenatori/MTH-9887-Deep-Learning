"""Minimal client for the WRDS data API (https://wrds-api.wharton.upenn.edu/data/).

The token is read from the environment variable WRDS_API_TOKEN or from a line
``WRDS_API_TOKEN=...`` in ``~/.env``. It is never logged and never written to
disk by this module.

Three properties of the API that the code relies on (checked 2026-11-21):

* ``count`` in a response is an estimate and can be wrong; only ``next`` says
  whether more rows follow.
* ``next`` is an ``http://`` URL. It is rewritten to ``https://`` before use, so
  the token never travels over an unencrypted connection.
* The server silently ignores a filter it does not support for a table (for
  comp.funda it ignored ``indfmt``, ``datafmt`` and the ``datadate`` range). Every
  filter is therefore applied again locally; ``Wrds.dropped`` counts the rows
  that the server sent although they did not match.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests

BASE = "https://wrds-api.wharton.upenn.edu/data/"


def load_token() -> str:
    tok = os.environ.get("WRDS_API_TOKEN")
    if tok:
        return tok.strip()
    env = Path.home() / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("WRDS_API_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(
        "WRDS_API_TOKEN is not set. Export it or put WRDS_API_TOKEN=<token> in ~/.env"
    )


def _eq(x, v) -> bool:
    if x is None:
        return False
    if str(x) == str(v):
        return True
    try:
        return float(x) == float(v)
    except (TypeError, ValueError):
        return False


def _matches(row: dict, key: str, val) -> bool:
    """Does ``row`` satisfy the query filter ``key=val`` (Django-style suffixes)?"""
    col, _, op = key.partition("__")
    if col not in row:
        return True
    x = row[col]
    if op == "":
        return _eq(x, val)
    if op == "in":
        return any(_eq(x, v) for v in str(val).split(","))
    if op in ("gte", "lte", "gt", "lt"):
        if x is None:
            return False
        a, b = str(x), str(val)          # ISO dates compare correctly as strings
        try:
            a, b = float(a), float(b)
        except ValueError:
            pass
        return {"gte": a >= b, "lte": a <= b, "gt": a > b, "lt": a < b}[op]
    return True


class Wrds:
    """Thin wrapper: one method, ``rows``, that returns every matching row."""

    def __init__(self, token: str | None = None, timeout: int = 240, max_retries: int = 7):
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": "Token " + (token or load_token()), "Accept": "application/json"}
        )
        self.timeout = timeout
        self.max_retries = max_retries
        self.dropped = 0      # rows the server sent that did not match the filters

    def _get(self, url: str, params: dict | None):
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (401, 403):
                    raise PermissionError(f"WRDS API refused the request ({r.status_code}) for {url}")
                if r.status_code == 404:
                    raise FileNotFoundError(f"WRDS API: no such table: {url}")
                # 429 and 5xx: wait and retry
            except (requests.ConnectionError, requests.Timeout, ValueError):
                pass
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        raise RuntimeError(f"WRDS API: giving up on {url} after {self.max_retries} attempts")

    def rows(self, table: str, params: dict, keep: list[str] | None = None, limit: int = 20000) -> list[dict]:
        """All rows of ``table`` (for example ``crsp.dsf``) that match ``params``.

        ``keep`` restricts the returned columns (the API always sends every
        column; the restriction only saves memory).
        """
        url: str | None = BASE + table + "/"
        filters = dict(params)
        p: dict | None = dict(params)
        p["limit"] = limit
        out: list[dict] = []
        while url:
            d = self._get(url, p)
            res = d["results"]
            n_all = len(res)
            res = [x for x in res if all(_matches(x, k, v) for k, v in filters.items())]
            self.dropped += n_all - len(res)
            if keep is not None:
                res = [{k: x.get(k) for k in keep} for x in res]
            out.extend(res)
            url = d.get("next")
            p = None  # the next URL carries its own query string
            if url and url.startswith("http://"):
                url = "https://" + url[len("http://"):]
        return out
