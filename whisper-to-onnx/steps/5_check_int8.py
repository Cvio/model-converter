"""Step 5: choose int8 files that still transcribe sensibly.

int8 roughly quarters the size, and cnverc's other models are int8. Its
transcript won't be identical to fp32's and doesn't need to be, but it must
still be sensible.

1. The export script's own int8 files (one scale per whole weight tensor).
2. If those aren't sensible, a per-channel requantization of the fp32 files
   (one scale per output channel), the same size, which usually holds up
   better on smaller models.
3. If neither is, stop. Setting use_fp32: true ships the full-size files.

Every candidate is transcribed with the engine cnverc runs, the same way as
the fp32 reference, so only the quantization differs.
"""

from common import (
    Stop,
    args,
    heading,
    load_config,
    main,
    read_json,
    run_dir,
    similarity,
    test_wav,
    transcribe_like_cnverc,
    write_json,
)

# int8 may change a word or two on a short clip. Below this, it is broken.
SENSIBLE = 0.85


def total_mb(*paths) -> float:
    return sum(p.stat().st_size for p in paths if p.is_file()) / 1e6


def step() -> None:
    a = args(__doc__)
    config = load_config(a.config)
    run = run_dir(config)
    onnx = run / "onnx"
    name = config["run_name"]
    read_json(onnx / "fp32.json", "step 4")
    wav = test_wav(config)
    language = config["language"]
    tokens = onnx / f"{name}-tokens.txt"

    fp32_files = (onnx / f"{name}-encoder.onnx", onnx / f"{name}-decoder.onnx")
    fp32_mb = total_mb(
        *fp32_files, onnx / f"{name}-encoder.weights", onnx / f"{name}-decoder.weights"
    )

    if config.get("use_fp32"):
        print("use_fp32 is set in the config: step 6 will ship the full-size files.")
        write_json(
            onnx / "chosen.json",
            {"method": "fp32", "encoder": fp32_files[0].name, "decoder": fp32_files[1].name},
        )
        return

    heading("The reference: fp32, in cnverc's engine")
    reference = transcribe_like_cnverc(*fp32_files, tokens, wav, language)
    print(f"  {reference}")

    def judge(label: str, encoder, decoder) -> bool:
        text = transcribe_like_cnverc(encoder, decoder, tokens, wav, language)
        agreement = similarity(reference, text)
        ok = bool(text.strip()) and agreement >= SENSIBLE
        print(f"  {text}")
        print(
            f"  {agreement:.0%} agreement with fp32, {total_mb(encoder, decoder):,.0f} MB "
            f"(fp32 is {fp32_mb:,.0f} MB): " + ("sensible" if ok else "not sensible")
        )
        results[label] = {"text": text, "agreement_with_fp32": agreement, "ok": ok}
        return ok

    results = {"fp32": {"text": reference}}
    chosen = None

    heading("1. The export script's int8 files (per-tensor)")
    script_int8 = (onnx / f"{name}-encoder.int8.onnx", onnx / f"{name}-decoder.int8.onnx")
    if judge("per-tensor", *script_int8):
        chosen = ("per-tensor", *script_int8)

    if chosen is None:
        heading("2. Requantizing per channel")
        from onnxruntime.quantization import QuantType, quantize_dynamic

        per_channel = (
            onnx / f"{name}-encoder.int8-per-channel.onnx",
            onnx / f"{name}-decoder.int8-per-channel.onnx",
        )
        for source, target in zip(fp32_files, per_channel):
            quantize_dynamic(
                model_input=str(source),
                model_output=str(target),
                op_types_to_quantize=["MatMul"],
                weight_type=QuantType.QInt8,
                per_channel=True,
            )
            print(f"  wrote {target.name}")
        if judge("per-channel", *per_channel):
            chosen = ("per-channel", *per_channel)

    write_json(onnx / "int8.json", results)
    if chosen is None:
        raise Stop(
            f"no int8 version transcribes sensibly where fp32 does. Set use_fp32: true in "
            f"{config['_path']} to ship the full-size files ({fp32_mb:,.0f} MB), then run "
            f"step 5 again."
        )
    method, encoder, decoder = chosen
    write_json(
        onnx / "chosen.json", {"method": method, "encoder": encoder.name, "decoder": decoder.name}
    )
    print(f"\nOK. Step 6 will ship the {method} int8 files.")
    print(
        "  One short clip is a thin basis. Step 7's comparison inside cnverc, on real "
        "speech, is the real judge."
    )


if __name__ == "__main__":
    main(step)
