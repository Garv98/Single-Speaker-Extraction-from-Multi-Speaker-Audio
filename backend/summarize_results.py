import json
r = json.load(open("paper_results.json"))

print("=== Per-mixture (real LibriSpeech) ===")
for rec in r["per_mixture"]:
    print(f"\n  mix {rec['index']}: input mix vs s1 = {rec['mixture_si_sdr_vs_s1_db']:.2f} dB, vs s2 = {rec['mixture_si_sdr_vs_s2_db']:.2f} dB")
    for name, e in rec["pipelines"].items():
        if "error" in e:
            print(f"    {name}: ERROR {e['error']}")
            continue
        line = f"    {name:24s} SISDR={e['si_sdr_db']:6.2f} | SISDRi={e['si_sdri_db']:6.2f} | matched={e['matched_speaker']}"
        if "icr_iterations" in e:
            line += f" | ICR={e['icr_iterations']} | sim={e['target_similarity']:.3f} | margin={e['confidence_margin']:.3f} | E_t/E_mix={e.get('energy_ratio_target',0):.3f}"
        print(line)

print("\n\n=== AGGREGATES ===")
for name, agg in r["aggregates"].items():
    if agg.get("n", 0) == 0:
        continue
    print(f"\n{name}  (n={agg['n']})")
    for k, v in agg.items():
        if k == "n":
            continue
        if isinstance(v, float):
            print(f"  {k:32s} {v:.4f}")
        else:
            print(f"  {k:32s} {v}")

print("\n\n=== ICR alpha trace summary (ecw_tse_with_ref) ===")
for rec in r["per_mixture"]:
    e = rec["pipelines"].get("ecw_tse_with_ref", {})
    if "icr_alpha_trace" in e:
        trace = e["icr_alpha_trace"]
        print(f"  mix {rec['index']}: trace = {[round(t,3) if t else None for t in trace]}")
