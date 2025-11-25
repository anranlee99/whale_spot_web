import { useEffect, useMemo, useState, type ChangeEvent } from "react";
import * as ort from "onnxruntime-web";
import "./index.css";
import whaleModelUrl from "./assets/WhaleBinary.onnx";

type LoadState = "idle" | "loading" | "ready" | "error";
type WhaleModel = {
  session: ort.InferenceSession | null;
  status: LoadState;
  error: string | null;
  modelHref: string;
  inputName: string | null;
};

// Bun dev returns asset URLs with the custom bun:// scheme; normalize to an http(s) URL so the browser can fetch it.
const normalizeAssetUrl = (assetUrl: string) => {
  try {
    const parsed = new URL(assetUrl);
    if (parsed.protocol === "bun:") {
      return new URL(parsed.pathname, window.location.origin).href;
    }
    return parsed.href;
  } catch {
    return assetUrl;
  }
};

function useWhaleModel(): WhaleModel {
  const modelHref = useMemo(() => normalizeAssetUrl(whaleModelUrl), []);
  const [status, setStatus] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [inputName, setInputName] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const loadModel = async () => {
      setStatus("loading");
      setError(null);

      try {
        // Point wasm assets to the CDN so the runtime can fetch the necessary binaries.
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";

        const createdSession = await ort.InferenceSession.create(modelHref, {
          executionProviders: ["wasm"],
        });

        if (cancelled) return;
        setSession(createdSession);
        setInputName(createdSession.inputNames[0] ?? null);
        setStatus("ready");
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
        setStatus("error");
      }
    };

    loadModel();

    return () => {
      cancelled = true;
    };
  }, [modelHref]);

  return { session, status, error, modelHref, inputName };
}

function ModelLoader({ model }: { model: WhaleModel }) {
  const { status, modelHref, error, session } = model;

  return (
    <div className="model-loader">
      <h2>ONNX Model</h2>
      <p className="model-path">Model file: {modelHref}</p>
      <div className="model-actions">
        {status === "ready" && session && <span className="status ready">Session ready for inference</span>}
        {status === "idle" && <span className="status idle">Not loaded</span>}
        {status === "loading" && <span className="status loading">Preloading model...</span>}
        {status === "error" && <span className="status error">Failed to load</span>}
      </div>
      {error && (
        <pre className="model-error" aria-live="polite">
          {error}
        </pre>
      )}
    </div>
  );
}

const IMAGE_SIZE = 328;
const CHANNEL_MEAN = [0.485, 0.456, 0.406];
const CHANNEL_STD = [0.229, 0.224, 0.225];

async function imageFileToTensor(file: File): Promise<ort.Tensor> {
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  canvas.width = IMAGE_SIZE;
  canvas.height = IMAGE_SIZE;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Canvas unsupported in this browser.");
  }
  ctx.drawImage(bitmap, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

  const imageData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  const { data } = imageData;

  const floatData = new Float32Array(1 * 3 * IMAGE_SIZE * IMAGE_SIZE);
  // Convert to NCHW with ImageNet normalization
  for (let y = 0; y < IMAGE_SIZE; y++) {
    for (let x = 0; x < IMAGE_SIZE; x++) {
      const idx = (y * IMAGE_SIZE + x) * 4;
      const r = data[idx] / 255;
      const g = data[idx + 1] / 255;
      const b = data[idx + 2] / 255;

      const base = y * IMAGE_SIZE + x;
      floatData[0 * IMAGE_SIZE * IMAGE_SIZE + base] = (r - CHANNEL_MEAN[0]) / CHANNEL_STD[0];
      floatData[1 * IMAGE_SIZE * IMAGE_SIZE + base] = (g - CHANNEL_MEAN[1]) / CHANNEL_STD[1];
      floatData[2 * IMAGE_SIZE * IMAGE_SIZE + base] = (b - CHANNEL_MEAN[2]) / CHANNEL_STD[2];
    }
  }

  return new ort.Tensor("float32", floatData, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
}

function WhalePredictor({ model }: { model: WhaleModel }) {
  const { session, status, inputName } = model;
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<{ label: string; prob: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const onFileChange = (ev: ChangeEvent<HTMLInputElement>) => {
    const file = ev.target.files?.[0];
    if (!file) return;
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setResult(null);
    setError(null);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const runInference = async () => {
    if (!session || !selectedFile || !inputName) {
      setError("Model not ready or no file selected.");
      return;
    }

    setRunning(true);
    setError(null);
    setResult(null);

    try {
      const tensor = await imageFileToTensor(selectedFile);
      const feeds: Record<string, ort.Tensor> = { [inputName]: tensor };
      const outputs = await session.run(feeds);
      const firstOutputName = session.outputNames[0];
      const outputTensor = outputs[firstOutputName as string];
      if (!outputTensor) {
        throw new Error("No output returned by the model.");
      }
      const raw = Array.isArray(outputTensor.data) ? outputTensor.data[0] : (outputTensor.data as Float32Array | number[])[0];
      const prob = 1 / (1 + Math.exp(-Number(raw))); // sigmoid
      const label = prob > 0.5 ? "WHALE" : "NO WHALE";
      setResult({ label, prob });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="predictor">
      <h2>Predict Whale Presence</h2>
      <p className="predictor-hint">Upload an image; we will resize to 328×328, normalize (ImageNet mean/std), and run the ONNX model.</p>

      <div className="predictor-row">
        <input type="file" accept="image/*" onChange={onFileChange} />
        <button type="button" onClick={runInference} disabled={!selectedFile || status !== "ready" || running}>
          {running ? "Running..." : "Run inference"}
        </button>
        <span className={`status ${status}`}>{status === "ready" ? "Model ready" : status === "loading" ? "Loading model..." : status}</span>
      </div>

      {previewUrl && (
        <div className="preview">
          <img src={previewUrl} alt="Selected for inference" />
        </div>
      )}

      {result && (
        <div className="result">
          <div className="label">{result.label}</div>
          <div className="prob">Confidence: {result.prob.toFixed(4)}</div>
        </div>
      )}

      {error && (
        <pre className="model-error" aria-live="polite">
          {error}
        </pre>
      )}
    </div>
  );
}

export function App() {
  const model = useWhaleModel();

  return (
    <div className="app">
      <WhalePredictor model={model} />
    </div>
  );
}

export default App;
