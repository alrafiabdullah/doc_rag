import { useEffect, useMemo, useState } from "react";
import type { ComponentProps } from "react";
import "./App.css";
import NotFoundPage from "./pages/NotFoundPage";
import ServerErrorPage from "./pages/ServerErrorPage";

type Source = {
	rank: number;
	source: string;
	preview: string;
};

type QueryMeta = {
	retrieved_chunks?: number;
	question_chars?: number;
	accepted_file_types?: string[];
	file_persisted?: boolean;
	generated_tokens?: number;
	generation_seconds?: number;
	tokens_per_second?: number;
};

type QueryResponse = {
	answer: string;
	meta?: QueryMeta;
	sources?: Source[];
};

type FormSubmitHandler = NonNullable<ComponentProps<"form">["onSubmit"]>;

const MAX_QUESTION_CHARS = 1_000;
const QUESTION_WARN_THRESHOLD = Math.floor(MAX_QUESTION_CHARS * 0.2);
const MAX_FILE_SIZE_BYTES = parseFloat(import.meta.env.VITE_MAX_FILE_SIZE_MB) * 1024 * 1024;
const STREAM_META_TAG = "[STREAM_META]";
const HF_TOKEN_STORAGE_KEY = "hf_token";
const RATE_LIMIT_MESSAGE = "Rate limit: 10 requests per 5 minutes.";
const DEFAULT_API_URL =
	import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const SERVER_ERROR_PATH = "/err";

function navigateToServerError() {
	if (typeof window === "undefined") {
		return;
	}

	if (window.location.pathname !== SERVER_ERROR_PATH) {
		window.location.assign(SERVER_ERROR_PATH);
	}
}

function App() {
	const [hfToken, setHfToken] = useState(() => {
		if (typeof window === "undefined") {
			return "";
		}
		return localStorage.getItem(HF_TOKEN_STORAGE_KEY) ?? "";
	});
	const [question, setQuestion] = useState("");
	const [topK, setTopK] = useState(3);
	const [stream, setStream] = useState(false);
	const [file, setFile] = useState<File | null>(null);

	const [loading, setLoading] = useState(false);
	const [answer, setAnswer] = useState("");
	const [sources, setSources] = useState<Source[]>([]);
	const [meta, setMeta] = useState<QueryMeta | null>(null);
	const [error, setError] = useState<string | null>(null);

	const pathname = typeof window !== "undefined" ? window.location.pathname : "/";
	const isServerErrorRoute = pathname === SERVER_ERROR_PATH;
	const isHomeRoute = pathname === "/" || pathname === "";

	useEffect(() => {
		if (isServerErrorRoute) {
			return;
		}

		const handleGlobalError = () => {
			navigateToServerError();
		};

		window.addEventListener("error", handleGlobalError);
		window.addEventListener("unhandledrejection", handleGlobalError);

		return () => {
			window.removeEventListener("error", handleGlobalError);
			window.removeEventListener("unhandledrejection", handleGlobalError);
		};
	}, [isServerErrorRoute]);

	const charsLeft = useMemo(
		() => MAX_QUESTION_CHARS - question.length,
		[question.length],
	);

	const canSubmit =
		hfToken.trim().length > 0 &&
		!!file &&
		question.trim().length > 0 &&
		question.length <= MAX_QUESTION_CHARS &&
		!loading;

	const handleTokenChange = (value: string) => {
		setHfToken(value);
		if (typeof window !== "undefined") {
			localStorage.setItem(HF_TOKEN_STORAGE_KEY, value);
		}
	};

	const handleFileChange = (nextFile: File | null) => {
		if (!nextFile) {
			setFile(null);
			return;
		}

		if (nextFile.size > MAX_FILE_SIZE_BYTES) {
			setFile(null);
			setError(`File size exceeds ${import.meta.env.VITE_MAX_FILE_SIZE_MB}MB limit.`);
			return;
		}

		setError(null);
		setFile(nextFile);
	};

	const handleSubmit: FormSubmitHandler = async (event) => {
		event.preventDefault();

		if (!file) {
			setError("Please select a .txt or .pdf file.");
			return;
		}

		if (file.size > MAX_FILE_SIZE_BYTES) {
			setError(`File size exceeds ${import.meta.env.VITE_MAX_FILE_SIZE_MB}MB limit.`);
			return;
		}

		if (!question.trim()) {
			setError("Please enter a question.");
			return;
		}

		if (!hfToken.trim()) {
			setError("Please enter your Hugging Face token.");
			return;
		}

		if (question.length > MAX_QUESTION_CHARS) {
			setError(`Question cannot exceed ${MAX_QUESTION_CHARS} characters.`);
			return;
		}

		setLoading(true);
		setError(null);
		setAnswer("");
		setSources([]);
		setMeta(null);

		try {
			const formData = new FormData();
			formData.append("file", file);
			formData.append("question", question);
			formData.append("top_k", String(topK));
			formData.append("stream", String(stream));

			const response = await fetch(`${DEFAULT_API_URL}/rag/query`, {
				method: "POST",
				headers: {
					Authorization: `Bearer ${hfToken.trim()}`,
				},
				body: formData,
			});

			if (!response.ok) {
				const maybeError = await response.json().then((data) => data.detail);
				throw new Error(maybeError || `Request failed (${response.status})`);
			}

			if (stream) {
				if (!response.body) {
					throw new Error("Streaming not supported in this browser.");
				}

				const reader = response.body.getReader();
				const decoder = new TextDecoder();
				let done = false;
				let buffer = "";
				let answerText = "";

				while (!done) {
					const result = await reader.read();
					done = result.done;

					if (result.value) {
						buffer += decoder.decode(result.value, { stream: true });

						const markerIndex = buffer.indexOf(STREAM_META_TAG);
						if (markerIndex >= 0) {
							const visible = buffer.slice(0, markerIndex);
							if (visible) {
								answerText += visible;
								setAnswer(answerText.trim());
							}

							const jsonPart = buffer
								.slice(markerIndex + STREAM_META_TAG.length)
								.trim();
							if (jsonPart) {
								try {
									const parsed = JSON.parse(jsonPart) as QueryMeta;
									setMeta(parsed);
									buffer = "";
								} catch {
									buffer = `${STREAM_META_TAG}${jsonPart}`;
								}
							} else {
								buffer = STREAM_META_TAG;
							}
						} else {
							answerText += buffer;
							setAnswer(answerText.trim());
							buffer = "";
						}
					}
				}

				const tail = buffer.trim();
				if (tail) {
					if (tail.startsWith(STREAM_META_TAG)) {
						const jsonPart = tail.slice(STREAM_META_TAG.length).trim();
						if (jsonPart) {
							try {
								const parsed = JSON.parse(jsonPart) as QueryMeta;
								setMeta(parsed);
							} catch {
								answerText += `\n\n${tail}`;
							}
						}
					} else {
						answerText += tail;
					}
					setAnswer(answerText.trim());
				}
			} else {
				const data = (await response.json()) as QueryResponse;
				setAnswer(data.answer || "");
				setSources(data.sources ?? []);
				setMeta(data.meta ?? null);
			}
		} catch (err) {
			const message = err instanceof Error ? err.message : "Request failed.";
			setError(message);
		} finally {
			setLoading(false);
		}
	};

	if (isServerErrorRoute) {
		return <ServerErrorPage />;
	}

	if (!isHomeRoute) {
		return <NotFoundPage />;
	}

	return (
		<main className="page">
			<section className="card hero-card">
				<p className="eyebrow">RAG Frontend</p>
				<h1>Upload, Ask, and Stream Answers</h1>
				<p className="subtitle">
					Accepts{" "}
					<strong>.txt</strong> and <strong>.pdf</strong>, supports streamed
					responses, and shows generation speed. Portfolio: {" "}
					<a href="https://abdullahalrafi.com?ti=rag" target="_blank" rel="noreferrer">
						abdullahalrafi.com
					</a>
				</p>
			</section>

			<section className="card">
				<form className="query-form" onSubmit={handleSubmit}>
					<p className="info-banner">{RATE_LIMIT_MESSAGE}</p>

					<label className="field">
						<span>Hugging Face Token</span>
						<input
							value={hfToken}
							onChange={(event) => handleTokenChange(event.target.value)}
							placeholder="hf_xxx..."
							type="password"
							autoComplete="off"
							required
						/>
						<small>
							We don&apos;t store your token or document. Everything happens at runtime in memory. View this project on{" "}
							<a
								href="https://github.com/alrafiabdullah/doc_rag"
								target="_blank"
								rel="noreferrer"
							>
								GitHub
							</a>
							.
						</small>
					</label>

					<label className="field">
						<span>Document (.txt or .pdf, max {import.meta.env.VITE_MAX_FILE_SIZE_MB}MB)</span>
						<input
							type="file"
							accept=".txt,.pdf,application/pdf,text/plain"
							onChange={(event) =>
								handleFileChange(event.target.files?.[0] ?? null)
							}
							required
						/>
					</label>

					<label className="field">
						<span>Question</span>
						<textarea
							value={question}
							onChange={(event) => setQuestion(event.target.value)}
							maxLength={MAX_QUESTION_CHARS}
							placeholder="Ask anything about the uploaded document..."
							rows={5}
							required
						/>
						<small className={charsLeft < QUESTION_WARN_THRESHOLD ? "warn" : ""}>
							{charsLeft} characters left
						</small>
					</label>

					<div className="row">
						<label className="field">
							<span>Top K (1-10)</span>
							<input
								type="number"
								min={1}
								max={10}
								value={topK}
								onChange={(event) => setTopK(Number(event.target.value))}
							/>
						</label>

						<label className="toggle">
							<input
								type="checkbox"
								checked={stream}
								onChange={(event) => setStream(event.target.checked)}
							/>
							<span>Stream answer</span>
						</label>
					</div>

					<button className="submit" type="submit" disabled={!canSubmit}>
						{loading ? "Generating…" : "Ask RAG"}
					</button>
				</form>

				{error ? <p className="error">{error}</p> : null}
			</section>

			<section className="card result-card">
				<div className="result-header">
					<h2>Answer</h2>
					<span className="pill">Stream: {stream ? "on" : "off"}</span>
				</div>

				<article className="answer-box">
					{answer
						? answer
						: "Your answer will appear here after submitting a query."}
				</article>

				{meta ? (
					<div className="meta-grid">
						{typeof meta.tokens_per_second === "number" ? (
							<div className="meta-item">
								<span>Tokens/sec</span>
								<strong>{meta.tokens_per_second.toFixed(2)}</strong>
							</div>
						) : null}
						{typeof meta.generated_tokens === "number" ? (
							<div className="meta-item">
								<span>Generated tokens</span>
								<strong>{meta.generated_tokens}</strong>
							</div>
						) : null}
						{typeof meta.generation_seconds === "number" ? (
							<div className="meta-item">
								<span>Generation time</span>
								<strong>{meta.generation_seconds.toFixed(2)}s</strong>
							</div>
						) : null}
						{typeof meta.retrieved_chunks === "number" ? (
							<div className="meta-item">
								<span>Retrieved chunks</span>
								<strong>{meta.retrieved_chunks}</strong>
							</div>
						) : null}
					</div>
				) : null}

				{sources.length > 0 ? (
					<div className="sources">
						<h3>Sources</h3>
						<ul>
							{sources.map((src) => (
								<li key={`${src.rank}-${src.source}`}>
									<div className="source-title">
										#{src.rank} • {src.source}
									</div>
									<p>{src.preview}</p>
								</li>
							))}
						</ul>
					</div>
				) : null}
			</section>
		</main>
	);
}

export default App;
