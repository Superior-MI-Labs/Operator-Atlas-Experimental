# Security, privacy, and supported deployment scope

**Operator Atlas is a localhost research application, not an authenticated hosted service.** Do not bind the bundled HTTP service to public or untrusted interfaces. There is no production-grade access control, multi-user isolation, DoS protection, sandbox for untrusted code, or independent security audit.

The research engine can store imported evidence in `data/evidence.sqlite3`. This and the runtime `logs/` directory are local private state and must not be committed. The release packager intentionally excludes them. Doctor collects bounded environment and study metadata in a ZIP; review it before public upload, especially when describing filesystem paths or hardware configuration.

For vulnerabilities, avoid posting working exploits or personal data in public issues. Contact the project maintainers privately through the Superior MI Labs organization contact channels if available. This repository does not promise a formal response SLA.
