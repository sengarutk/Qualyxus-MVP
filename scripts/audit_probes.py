import sys, json, time, queue, threading, asyncio, os
sys.path.insert(0,"/home/sengar/dev"); sys.path.insert(0,"/home/sengar/dev/tests")
import numpy as np, pytest, cv2
from pathlib import Path
from test_model_ops import _reject_path_manager, _StubStack, _health_fields
from src.camera.part_inspection import *
from src.process.synthetic_parts import render_part
from src.process.simulator import InjectionMoldingSimulator
from src.process.models import *

def test_01_failed_real_camera_must_not_become_synthetic():
    import yaml
    c=yaml.safe_load(open("/home/sengar/dev/config.yaml"))
    on=[x["id"] for x in c["cameras"] if x.get("fallback_to_synthetic_on_failure")]
    src=open("/home/sengar/dev/src/camera/camera_manager.py").read()
    assert "transitioning immediately to synthetic" not in src, f"code path exists; enabled in shipped config for {on}"

def test_02_empty_textured_station_is_not_a_part():
    from src.camera.camera_manager import is_part_present
    bg=np.tile(np.linspace(60,110,640,dtype=np.uint8),(480,1)); bg=cv2.merge([bg]*3)
    assert not is_part_present(bg), "an empty station with ordinary uneven lighting counts as a part present"
def test_03_parts_are_found_on_uneven_lighting():
    from src.camera.camera_manager import is_part_present
    t=PartPresenceTracker("c"); n=0
    bg=cv2.merge([np.tile(np.linspace(60,110,640,dtype=np.uint8),(480,1))]*3)
    for k in range(10):
        for j in range(10): 
            r=t.update(bg,is_part_present(bg),k*2+j*.1,j); n+=r is not None
        p=bg.copy(); cv2.rectangle(p,(200,150),(440,330),(170,170,170),-1)
        for j in range(10):
            r=t.update(p,is_part_present(p),k*2+1+j*.1,j); n+=r is not None
    assert n>=9, f"{n} of 10 parts detected"

def test_04_frame_pool_memory_for_12mp():
    from src.camera.camera_manager import _POOL_SLOTS
    gb=_POOL_SLOTS*4000*3000*3/1e9; assert gb<1, f"{gb:.1f} GB RAM per 12 MP camera, {6*gb:.0f} GB for six"

def test_05_two_cameras_without_required_view_must_not_overwrite_each_other():
    f=PartVerdictFuser(); a=f.add("m-1","cam_0",3,0.); b=f.add("m-1","cam_1",0,.1)
    assert b is None or b["verdict"]!="pass", "cam_0 rejected the part, then cam_1 published PASS for the same part"
def test_06_shipped_config_multi_camera_has_required_views():
    import yaml; c=yaml.safe_load(open("/home/sengar/dev/config.yaml"))["cameras"]
    en=[x for x in c if x.get("enabled",True)]; assert len(en)<2 or all(x.get("required_view") for x in en), f"{len(en)} cameras, none required"

def test_07_part_number_changes_with_every_verdict(tmp_path):
    cm,st=_reject_path_manager(tmp_path,_StubStack(False,"v",.4)); cm.part_fuser=PartVerdictFuser(["a"],timeout_s=5); nums=[]
    for key in ["m-1","m-2","m-3",None,"m-4","m-5"]:
        st.camera_id="a"; cm._report_view(st,key,0)
    while not cm.fast_queue.empty(): nums.append(cm.fast_queue.get()["part"])
    assert all(x!=y for x,y in zip(nums,nums[1:])), f"PLC sees {nums}: repeated number = verdict it cannot notice"

def test_08_stale_verdict_is_not_written_late(tmp_path):
    cm,_=_reject_path_manager(tmp_path,_StubStack(False,"v",.4)); wrote=[]
    class A:
        def write_part_verdict(s,v): wrote.append(v); return True
    cm._deliver_verdict(A(),{"part":1,"flag":0,"verdict":"pass","timestamp":time.time()-60})
    assert not wrote, "a verdict 60 s old is still written to the PLC"

def test_09_kpi_counts_parts_not_views(tmp_path):
    from src.process.store import ProcessStore; from src.process.kpis import compute_kpis
    s=ProcessStore(tmp_path/"p.db")
    for i in range(10):
        for v in range(6): s.save_inspection(InspectionRecord("",f"v{v}",100+i,FAIL if (i==0 and v==0) else PASS,cycle_id=f"m-{i}",alignment_confidence=.9))
    k=compute_kpis(s,0,1e4); assert k["parts_inspected"]==10 and abs(k["scrap_rate"]-.1)<1e-6, f"10 parts, 1 reject (10% scrap) shown as {k['parts_inspected']} parts, {k['scrap_rate']:.1%}"
def test_10_not_inspected_not_counted_good(tmp_path):
    from src.process.store import ProcessStore; from src.process.kpis import compute_kpis
    s=ProcessStore(tmp_path/"p.db")
    for i in range(10): s.save_inspection(InspectionRecord("","v",100+i,"not_inspected" if i<9 else FAIL,cycle_id=f"m-{i}"))
    k=compute_kpis(s,0,1e4); assert k["scrap_rate"]==1.0, f"1 judged part, rejected; shown scrap {k['scrap_rate']:.0%}"
def test_11_incomplete_part_is_not_a_good_label(tmp_path):
    from src.process.store import ProcessStore
    s=ProcessStore(tmp_path/"p.db"); c,_=InjectionMoldingSimulator().next_cycle(); s.save_cycle(c,{"a":1.0})
    s.save_inspection(InspectionRecord("","top",1.,PASS,cycle_id=c.cycle_id,alignment_confidence=.9)); s.save_inspection(InspectionRecord("","side",1.,"not_inspected",cycle_id=c.cycle_id,alignment_confidence=.9))
    assert s.labelled_dataset()==[], "part with an uninspected view trains the risk model as good"
def test_12_retrain_frequency_counts_parts(tmp_path):
    from src.process.service import ProcessIntelligenceService
    svc=ProcessIntelligenceService({"db_path":str(tmp_path/"p.db"),"models_dir":str(tmp_path/"m"),"model":{"retrain_every_labels":10**9}})
    c,_=InjectionMoldingSimulator().next_cycle(); svc.on_cycle(c)
    from src.process.alignment import AlignmentResult
    for v in range(6): svc.record_inspection(f"v{v}",1.,PASS,alignment=AlignmentResult(c.cycle_id,.9,"time"))
    assert svc._labels_since_train==1, f"one six-view part counts as {svc._labels_since_train} labels: retrains every 42 parts instead of 250"
def test_13_lost_cycles_are_reported():
    from src.process.sources import CycleAssembler,SignalSpec
    a=CycleAssembler("m",[SignalSpec("p")]); out=[]
    for i in range(10): out.append(a.feed(i*.1,{"p":1.},100))
    for i in range(10): out.append(a.feed(5+i*.1,{"p":1.},105))     # 5 shots happened during a network outage
    c=[x for x in out if x]; assert not c or c[0].duration_s<2 and hasattr(a,"missed"), "4 machine cycles vanished with no record"
def test_14_outage_samples_are_not_merged_into_one_cycle():
    from src.process.sources import CycleAssembler,SignalSpec
    a=CycleAssembler("m",[SignalSpec("p")]); got=None
    for i in range(10): a.feed(i*.1,{"p":1.},100)
    for i in range(10): a.feed(60+i*.1,{"p":1.},100)
    got=a.feed(62,{"p":1.},101); assert got is None or got.duration_s<20, f"59 s outage swallowed into one {got.duration_s:.0f} s cycle"
def test_15_health_baseline_is_per_machine(tmp_path):
    import inspect; from src.process.store import ProcessStore
    assert "machine_id" in inspect.signature(ProcessStore.cycle_features).parameters, "cycles of all machines share one health baseline"
def test_16_health_baseline_resets_after_maintenance():
    src=open("/home/sengar/dev/src/console/data.py").read()+open("/home/sengar/dev/src/process/machine_health.py").read()
    assert "rebaseline" in src or "baseline_reset" in src, "baseline is the first 100 cycles ever recorded, for life"
def test_17_retrain_on_truncated_pool_keeps_false_reject_limit():
    from src.inference.anomaly.calibration import conformal_threshold
    r=np.random.default_rng(0); good=lambda n: r.lognormal(0,.25,n)
    t0=conformal_threshold(good(5000),.01)["threshold"]; pool=good(20000); pool=pool[pool<=.8*t0]
    t1=conformal_threshold(pool[:1500],.01)["threshold"]; frr=float((good(50000)>t1).mean())
    assert frr<=.015, f"threshold retrained from the good-part pool rejects {frr:.1%} of good parts (target 1%) while reporting a guarantee"
def test_18_retrained_model_keeps_view_budget():
    import inspect; from src.inference import model_registry as m
    assert "n_views" in inspect.getsource(m._retrain), "retraining ignores the number of views: six-view false rejects return"
def test_19_rollback_twice_never_returns_to_bad_model(tmp_path):
    from src.inference.model_registry import VisionModelRegistry as R
    reg=R(tmp_path/"r"); 
    for i in range(3): (tmp_path/f"m{i}.pt").write_text(str(i)); reg.register("a","anomaly",tmp_path/f"m{i}.pt",activate=True)
    seq=[reg.rollback("a","anomaly",by="x",reason="bad") for _ in range(2)]; assert seq==["v0002","v0001"], f"v0003 bad, v0002 bad -> second rollback gives {seq[1]}"
def test_20_registry_verifies_model_file(tmp_path):
    import inspect; from src.inference import inspection_stack as m
    assert "file_digest(" in inspect.getsource(m._load_anomaly), "digest is recorded but never checked at load"
def test_21_low_side_drift_alarms():
    from src.inference.anomaly.monitoring import ModelHealthMonitor
    m=ModelHealthMonitor(calibration={"holdout_mean":.5},threshold=1.)
    for _ in range(100): s=m.update(.05,False)
    assert s["alerts"], "scores collapse to a tenth of normal (model blind) with no alert"
@pytest.mark.parametrize("name,fn",[("20% darker",lambda f:(f*.8).astype(np.uint8)),("40% brighter",lambda f:np.clip(f*1.4,0,255).astype(np.uint8)),("glare half",lambda f:np.hstack([f[:,:f.shape[1]//2],np.full_like(f[:,f.shape[1]//2:],255)])),("half covered",lambda f:np.hstack([f[:,:f.shape[1]//2],np.zeros_like(f[:,f.shape[1]//2:])]))])
def test_22_lighting_fault_is_a_fault_not_a_verdict(name,fn):
    from src.inference.anomaly.image_quality import image_fault,image_quality_baseline
    base=image_quality_baseline([render_part(640,480,None,s) for s in range(10)])
    assert image_fault(fn(render_part(640,480,None,99)),base), name
def test_23_shutdown_delivers_last_verdicts():
    src=open("/home/sengar/dev/src/camera/camera_manager.py").read(); i=src.index("def _sync_stop"); b=src[i:i+2500]
    assert b.index("inspection_executor.shutdown")<b.index("_reflex_thread, self._disk") and "cancel_futures=True" not in b[b.index("inspection_executor"):b.index("inspection_executor")+70], "reflex thread stops before inspection; queued parts are cancelled without a verdict"
def test_24_config_typo_is_caught(tmp_path):
    from src.config.config import Config
    (tmp_path/"c.yaml").write_text("cameras:\n - {id: a, requred_view: true}\ninspection: {view_timout_s: 1}\n"); os.chdir(tmp_path)
    with pytest.raises(Exception): Config.load(config_path=str(tmp_path/"c.yaml"))
def test_25_event_batch_survives_cancel(tmp_path,monkeypatch):
    from src.logging.event_logger import EventLogger
    import sqlite3
    EventLogger._instance=None; l=EventLogger(str(tmp_path/"e.db"),str(tmp_path/"e.csv"))
    async def go():
        l.queue=asyncio.Queue(); await l.start(); l.timeout_sec=5
        l.queue.put_nowait({"camera_id":"c","box":[1],"track_id":1,"timestamp":1.0}); await asyncio.sleep(.2); l.worker_task.cancel()
        await asyncio.gather(l.worker_task,return_exceptions=True)
    asyncio.run(go()); n=sqlite3.connect(tmp_path/"e.db").execute("select count(*) from events").fetchone()[0]; EventLogger._instance=None
    assert n==1, "event already taken from the queue is lost at shutdown"
def test_26_telemetry_never_invents_values():
    from src.telemetry.plc_adapter import PLCAdapter
    a=PLCAdapter().read_context("cam_0"); assert a["lot_id"] is None, f"default config writes random data into event records: {a['temperature']} C, {a['lot_id']}, shift {a['shift_id']}"
def test_27_ai_errors_are_not_saved_as_findings():
    from src.inference.analysis.vlm_gateway import analyze_defect_crop_sync
    class C: raw={"vlm_reasoning":{"enabled":True,"local_api":"http://127.0.0.1:9"}}
    r=analyze_defect_crop_sync(np.zeros((8,8,3),np.uint8),"x",C); assert not r, f"stored as the defect's root cause: {r[:70]}"
def test_28_late_failing_view_is_not_ignored():
    f=PartVerdictFuser(["a","b"],timeout_s=5); f.add("m-1","a",0,0.); f.expired(6.); r=f.add("m-1","b",3,7.)
    assert r is not None, "a reject arriving after the timeout never reaches the PLC (part already flagged 4, not reject)"
def test_29_recipe_mismatch_is_not_judged_by_wrong_model(tmp_path):
    from src.inference.inspection_stack import StackProvider
    (tmp_path/"a.pt").write_bytes(b"x"); p=StackProvider({"anomaly_model":str(tmp_path/"a.pt"),"recipes":{"A":{"anomaly_model":None}}})
    calls=[]; import src.inference.inspection_stack as m; o=m.InspectionStack; m.InspectionStack=lambda r,**k:calls.append((r,k.get("anomaly_path"))) or type("X",(),{"versions":{}})()
    try: p.get("UNKNOWN_PRODUCT")
    finally: m.InspectionStack=o
    assert calls[0][1] is None, "an unknown product is judged with the default product's model"
def test_30_unnamed_duplicate_camera_ids_rejected(tmp_path):
    from src.config.config import Config
    (tmp_path/"c.yaml").write_text("cameras:\n - {id: a}\n - {id: a}\n"); os.chdir(tmp_path)
    with pytest.raises(Exception): Config.load(config_path=str(tmp_path/"c.yaml"))
def test_31_console_token_not_in_url():
    assert "token: str | None" not in open("/home/sengar/dev/src/console/server.py").read(), "token accepted in the URL (history, proxy logs)"
def test_32_removed_user_loses_access():
    src=open("/home/sengar/dev/src/dashboard/app.py").read(); i=src.index("session = st.session_state.get"); assert "users()" in src[i:i+500], "a removed or demoted user keeps the role for up to 8 hours"
def test_33_six_views_are_six_faces():
    a=[cv2.resize(cv2.cvtColor(render_part(640,480,None,5+i*7919),cv2.COLOR_BGR2GRAY),(64,48)).astype(float) for i in range(6)]
    c=[np.corrcoef(a[0].ravel(),x.ravel())[0,1] for x in a[1:]]; assert max(c)<.6, f"all six 'faces' in the demo are the same face (correlation {min(c):.2f}-{max(c):.2f})"
