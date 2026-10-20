# Security

Security fixes go to the latest release (1.0.x).

Report vulnerabilities privately through GitHub: Security > Report a vulnerability
on the repository. Please do not open public issues for them.

Memcat parses stored entries with `struct`, checks every length against the buffer
and a CRC32, and never uses `pickle` or `eval`. Treat your Redis or S3 store as
holding your agent's voice output: keep it private to your deployment.

`memcat serve` listens on 127.0.0.1 by default and refuses any other address
without a secret (`[serve] secret` or `MEMCAT_SERVE_SECRET`, sent as
`X-VAPI-SECRET`) unless you pass `--insecure-no-auth`. Without a secret it also
refuses requests from web pages (an `Origin` header) and with a non-loopback
`Host`. To erase audio, see
[Erasure](https://github.com/imumesh18/memcat/blob/main/docs/privacy.md#erasure).
