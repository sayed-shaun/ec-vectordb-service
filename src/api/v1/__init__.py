"""Version 1 of the HTTP surface.

`v1` is a source layout. The top-k route carries the version in its URL
(`/api/v1/topk`); `/reindex` and `/health` are unversioned operational routes,
and `/health` stays put for the container health check. When a breaking change
is needed, a `v2` package mounts alongside under an explicit prefix rather than
moving anything that already ships.
"""
