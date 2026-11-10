# Resilience Test Report

| Scenario | Failure Type | Outcome | Execution Time (s) | Clean Recovery |
|----------|--------------|---------|---------------------|----------------|
| camera_drop | camera_disconnect | pass | 10.91 | Yes |
| db_lock | database_lock | pass | 6.53 | Yes |
| disk_full | disk_exhaustion | pass | 4.02 | Yes |
| network_delay | network_latency | pass | 10.56 | Yes |
