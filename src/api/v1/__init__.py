"""Version 1 of the HTTP surface.

`v1` is a **source layout**, not a URL prefix. The routes are unversioned
(`/topk/top_similar/`, `/reindex`, `/health`). When a breaking change is
needed, a `v2` package mounts alongside under an explicit prefix rather than
moving anything that already ships.
"""
