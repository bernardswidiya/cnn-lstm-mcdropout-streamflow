"""Proses pekerja untuk jalankan_paralel di pemodelan.py.

Membaca daftar tugas (JSON), melatih setiap run yang belum ada di log, dan mencatatnya ke log proses ini
(nama file dari variabel lingkungan PM_LOG_RUN, misalnya log_run_p0.csv)."""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pemodelan as pm

assert pm.LOG_PROSES != pm.LOG_RUN, "pekerja harus menulis ke log proses sendiri (PM_LOG_RUN)"
tugas = json.loads(Path(sys.argv[1]).read_text())
print(f"{pm.mesin()}: {len(tugas)} tugas, log {pm.LOG_PROSES.name}", flush=True)
data, kunci_data = None, None
for i, t in enumerate(tugas, 1):
    k = (t["h"], t["varian"], t["jendela"], t["hujan"], t["transformasi"])
    if k != kunci_data:
        data, kunci_data = None, None
        data, kunci_data = pm.muat_latih_val(*k), k
    t0 = time.perf_counter()
    if t.get("finalis"):
        r = pm.jalankan_finalis(t["notebook"], t["tahap"], t["model"], t["cfg"], data, t["seed"])
    else:
        r = pm.jalankan(t["notebook"], t["tahap"], t["model"], t["cfg"], data, t["seed"])
    print(f"[{i}/{len(tugas)}] {t['tahap']} {t['model']} {t['varian']} w{t['jendela']} {t['hujan']} {t['transformasi']} "
          f"h{t['h']} s{t['seed']}: NSE {float(r['NSE']):.4f}, {time.perf_counter() - t0:.0f} s"
          f"{' (dari log)' if r['dari_log'] else ''}", flush=True)
