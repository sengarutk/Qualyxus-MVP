import os
import sys
import importlib
import traceback

def test_repository_integrity():
    print("Initiating full repository module linking test...")
    modules_to_test = [
        "src.config.config",
        "src.config.vault",
        "src.core.factory_manager",
        "src.core.hub_connector",
        "src.network.mqtt_bridge",
        "src.logging.event_logger",
        "src.logging.schema",
        "src.camera.camera_manager",
        "src.inference.infer_live",
        "src.inference.postprocess",
        "src.inference.tracker",
        "src.inference.analysis.vibration_engine",
        "src.system.disk_monitor",
        "src.system.resource_monitor",
        "src.system.runtime_metrics",
        "src.system.watchdog",
        "src.telemetry.telemetry_provider",
        "src.api.pulse_server",
        "main"
    ]
    
    failed = []
    
    # Pre-mock os.environ logic to prevent immediate failures during import
    os.environ['SQLITE_DB_PATH'] = 'data/test_events.db'
    os.environ['CSV_FALLBACK_PATH'] = 'data/test_events.csv'
    
    for mod in modules_to_test:
        try:
            importlib.import_module(mod)
            print(f"[OK] {mod}")
        except Exception as e:
            print(f"[FAILED] {mod}")
            traceback.print_exc()
            failed.append(mod)
            
    if failed:
        print(f"\nIntegration validation failed for {len(failed)} modules.")
        sys.exit(1)
    else:
        print("\nAll modules loaded successfully! ZERO broken links.")
        sys.exit(0)

if __name__ == "__main__":
    test_repository_integrity()
