"""Version 1 of the HTTP surface.

`v1` is a **source layout**, not a URL prefix. The routes stay unversioned
(`/ec_bot/top_similar/`, `/query`) because this service has to stand in for the
existing one at `:8002`, whose paths are fixed. When a breaking change is
needed, a `v2` package mounts alongside under an explicit prefix rather than
moving anything that already ships.
"""
