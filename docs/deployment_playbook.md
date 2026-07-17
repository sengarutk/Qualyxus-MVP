# QUALYXUS Industrial Deployment Playbook

## 1. Overview

QUALYXUS is an AI-powered manufacturing intelligence system that provides real-time defect detection, root cause analysis, and automated action tracking for production lines. It integrates computer vision with operational workflows to minimize downtime and improve quality control.

### Key Capabilities
- Real-time defect detection using YOLOv8 segmentation
- Automated root cause analysis with statistical correlation
- Action lifecycle management with approval workflows
- Production KPI monitoring and reporting
- Shift handover and audit trails

### Deployment Scenarios
- **Single-Line Deployment**: One production line with 1-4 cameras, suitable for pilot installations
- **Multi-Line Deployment**: Multiple production lines (up to 10+), centralized monitoring and control

## 2. Deployment Architecture

QUALYXUS follows a modular architecture designed for industrial reliability and scalability.

### System Components
- **Camera Layer**: Industrial cameras capture production line footage via RTSP streams
- **Inference Engine**: YOLOv8 models process video frames for defect detection
- **Analytics Pipeline**: Statistical analysis and root cause correlation
- **Action Management**: Workflow tracking with approval gates and audit trails
- **Dashboard Interface**: Real-time monitoring and reporting for operators

### Data Flow
```
Cameras → Inference → Events DB → RCA → Actions DB → Dashboard
     ↑           ↑           ↑         ↑         ↑         ↑
   RTSP      YOLOv8    SQLite    Stats    Tracker   Streamlit
```

### Scalability Considerations
- Horizontal scaling via multiple inference workers
- Database partitioning for high-volume deployments
- Load balancing for multi-line operations

## 3. Hardware Requirements

### Minimum Requirements
- **CPU**: 8-core processor (Intel i7/AMD Ryzen 7 or equivalent)
- **RAM**: 16GB DDR4
- **Storage**: 512GB SSD for OS and data
- **Network**: 1Gbps Ethernet
- **OS**: Ubuntu 20.04 LTS or Windows Server 2019+

### Recommended Specifications
- **CPU**: 12-core processor (Intel i9/AMD Ryzen 9)
- **RAM**: 32GB DDR4
- **Storage**: 1TB NVMe SSD + 2TB HDD for data retention
- **GPU**: NVIDIA RTX 3060 or higher (optional, improves inference speed)
- **Network**: 10Gbps Ethernet for multi-camera setups

### Industrial Camera Specifications
- **Resolution**: 1920x1080 (1080p) minimum, 4K preferred
- **Frame Rate**: 10-30 FPS
- **Protocol**: RTSP over IP, ONVIF compliant
- **Lighting**: IR-capable for low-light conditions
- **Mounting**: Vibration-resistant industrial enclosures

### Resource Sizing Guidelines

| Cameras | CPU Cores | RAM (GB) | GPU Memory | Storage (TB) | Network |
|---------|-----------|----------|------------|--------------|---------|
| 1-2 | 8 | 16 | 4GB | 0.5 | 1Gbps |
| 3-4 | 12 | 32 | 8GB | 1.0 | 1Gbps |
| 5-8 | 16 | 64 | 12GB | 2.0 | 10Gbps |
| 9-12 | 24 | 128 | 24GB | 4.0 | 10Gbps |

*Note: GPU optional but recommended for real-time performance. Storage includes 90-day retention.*

## 4. Network Setup

### Camera Networking
- Use dedicated VLAN for camera traffic
- RTSP streams: TCP ports 554 (standard) or 8554 (alternative)
- ONVIF discovery: UDP port 3702
- Ensure QoS prioritization for video streams

### Port Requirements
| Service | Protocol | Port | Direction |
|---------|----------|------|-----------|
| Dashboard | HTTP | 8501 | Inbound |
| RTSP Streams | TCP | 554/8554 | Inbound |
| Database | TCP | 5432 (if PostgreSQL) | Local |
| SSH | TCP | 22 | Inbound (admin only) |

### Firewall Rules
- Allow inbound RTSP from camera subnet
- Restrict dashboard access to operations VLAN
- Enable outbound for model updates and reporting
- Block all unused ports

## 4. Installation Steps

### Prerequisites
- Ubuntu 20.04 LTS server
- Python 3.9+
- Git client
- Administrative access

### Step-by-Step Installation

1. **Clone Repository**
   ```bash
   git clone https://github.com/your-org/qualyxus.git
   cd qualyxus
   ```

2. **Install System Dependencies**
   ```bash
   sudo apt update
   sudo apt install -y python3-pip python3-venv postgresql postgresql-contrib
   ```

3. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure System**
   ```bash
   cp config/default.yaml config/config.yaml
   # Edit config.yaml with site-specific settings
   nano config/config.yaml
   ```

6. **Initialize Database**
   ```bash
   # Create PostgreSQL database
   sudo -u postgres createdb qualyxus
   sudo -u postgres psql -c "CREATE USER qualyxus WITH PASSWORD 'secure_password';"
   sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE qualyxus TO qualyxus;"

   # Run database migrations
   python -m src.system.db_init
   ```

7. **Test Inference**
   ```bash
   python scripts/test_inference.py
   ```

8. **Launch Dashboard**
   ```bash
   python -m src.dashboard.app
   ```

## Pilot Go-Live Checklist

- [ ] Hardware verification: CPU, RAM, storage meet requirements
- [ ] Camera installation: All cameras mounted, powered, network connected
- [ ] Network configuration: VLANs, ports, firewall rules applied
- [ ] Software installation: Dependencies installed, configuration complete
- [ ] Database initialization: Schemas created, test data loaded
- [ ] Camera calibration: Test streams, focus, lighting verified
- [ ] Inference testing: Run test_inference.py, verify detections
- [ ] Dashboard access: Login credentials distributed, permissions set
- [ ] Operator training: Basic operation, reporting, handover procedures
- [ ] Backup verification: Database backup and restore tested
- [ ] Go-live dry run: 24-hour test operation with monitoring
- [ ] Emergency contacts: Support team notified, escalation procedures documented

## 5. Runtime Operations

### Monitoring
- Access dashboard at http://server:8501
- Monitor camera health, inference latency, and defect rates
- Set up alerts for camera disconnects or high defect rates

### Data Retention
- Raw images: 30 days
- Defect data: 90 days
- Logs: 180 days
- Configure retention policies in config.yaml

### Shift Handover
- Use dashboard shift handover feature
- Review open actions and pending issues
- Document handoff notes for incoming shift

### Reporting Workflow
- Generate compliance reports weekly
- Export audit trails monthly
- Review action effectiveness quarterly

## Security Considerations

### Access Control
- Implement role-based access control (RBAC) for dashboard users
- Use LDAP/Active Directory integration for enterprise authentication
- Enforce multi-factor authentication for administrative accounts
- Regularly rotate service account credentials

### Data Protection
- Encrypt sensitive data at rest using database encryption
- Use TLS 1.3 for all network communications
- Implement data masking for PII in logs and reports
- Comply with GDPR/CCPA for personal data handling

### Network Security
- Segment camera networks from operational networks
- Deploy intrusion detection systems (IDS) on critical segments
- Monitor for unauthorized access attempts
- Regular security patching and vulnerability assessments

### Operational Security
- Limit physical access to server hardware
- Implement secure backup procedures with encryption
- Monitor system logs for security events
- Conduct regular security awareness training for operators

## Operational SLAs

### System Availability
- **Uptime**: 99.9% monthly availability (8.77 hours downtime/year)
- **Maintenance Windows**: Scheduled for Sundays 02:00-04:00 UTC
- **Incident Response**: <4 hours for critical issues, <24 hours for high priority

### Performance Metrics
- **Inference Latency**: <500ms per frame (95th percentile)
- **Dashboard Response**: <2 seconds for standard queries
- **Camera Reconnect**: <30 seconds automatic recovery

### Data Integrity
- **Retention Compliance**: 100% adherence to configured policies
- **Backup Recovery**: <4 hours RTO, <1 hour RPO
- **Audit Trail**: Complete traceability for all actions and changes

## 6. Maintenance Guide

### Log Rotation
```bash
# Configure logrotate for application logs
sudo nano /etc/logrotate.d/qualyxus
/var/log/qualyxus/*.log {
    daily
    rotate 30
    compress
    missingok
}
```

### Database Backup
```bash
# Daily backup script
pg_dump qualyxus > /backup/qualyxus_$(date +%Y%m%d).sql
# Store backups for 90 days
find /backup -name "*.sql" -mtime +90 -delete
```

### System Updates
- Schedule maintenance windows for updates
- Test updates in staging environment first
- Update models quarterly or when new defect types are identified

### Retention Tuning
- Monitor storage usage weekly
- Adjust retention policies based on compliance requirements
- Archive old data to cold storage as needed

## 7. Troubleshooting

### Camera Disconnect
**Symptoms**: Camera shows offline in dashboard
**Causes**: Network issues, power failure, camera hardware fault
**Resolution**:
1. Check network connectivity to camera IP
2. Verify RTSP stream URL in config
3. Restart camera manager: `systemctl restart qualyxus-camera`
4. Check camera logs for connection errors

### Database Lock
**Symptoms**: System unresponsive, queries timeout
**Causes**: Long-running queries, deadlocks, insufficient connections
**Resolution**:
1. Check database connection pool settings
2. Kill long-running queries: `SELECT pg_cancel_backend(pid) FROM pg_stat_activity WHERE state = 'active' AND now() - query_start > interval '5 minutes';`
3. Restart application if necessary
4. Optimize slow queries

### Disk Full
**Symptoms**: Inference stops, dashboard shows storage alerts
**Causes**: Uncontrolled data growth, failed retention cleanup
**Resolution**:
1. Check disk usage: `df -h`
2. Run manual cleanup: `python -m src.system.data_retention`
3. Increase storage capacity
4. Review and adjust retention policies

### Inference Stall
**Symptoms**: Cameras online but no detections, high CPU usage
**Causes**: Model corruption, GPU memory issues, input stream problems
**Resolution**:
1. Check GPU memory: `nvidia-smi`
2. Restart inference workers: `systemctl restart qualyxus-inference`
3. Verify model files integrity
4. Check input stream quality and frame rates