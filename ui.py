import json
import logging
import os
from typing import Any, Optional

import altair as alt
import pandas as pd
import requests
import streamlit as st

logger = logging.getLogger("enterprise_rag_ui")


def _get_config_value(key: str, default: str = "") -> str:
    value = None

    if hasattr(st, "secrets"):
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None

    if value is None:
        value = os.getenv(key)

    return str(value).strip() if value is not None else default


def _default_api_base_url() -> str:
    return _get_config_value("RAG_API_URL", "http://localhost:8000").rstrip("/")


def _normalize_api_base_url(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return _default_api_base_url()

    if not (value.startswith("http://") or value.startswith("https://")):
        value = f"http://{value}"

    return value.rstrip("/")


def _require_auth() -> None:
    password = _get_config_value("UI_PASSWORD", "")
    if not password:
        return

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return

    st.title("Enterprise RAG")
    st.caption("Sign in to continue.")
    entered = st.text_input(
        "Password",
        type="password",
        help="Access is restricted. Contact the administrator if you don't have credentials.",
    )
    if st.button("Sign in", type="primary", use_container_width=True):
        if entered == password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid password")
    st.stop()


@st.cache_data(ttl=5)
def _cached_health(base_url: str, headers: dict[str, str], timeout_s: float) -> tuple[bool, Any]:
    """Cache health calls briefly to keep the UI responsive."""
    try:
        base_url = _normalize_api_base_url(base_url)
        r = requests.get(f"{base_url}/health", headers=headers, timeout=timeout_s)
        return r.ok, r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
    except Exception as e:
        return False, str(e)


def _init_session_state() -> None:
    """Initialize Streamlit session state defaults."""
    defaults = {
        "api_base_url": _default_api_base_url(),
        "timeout_s": float(_get_config_value("RAG_TIMEOUT_S", "30") or 30),
        "api_key_header": _get_config_value("RAG_API_KEY_HEADER", "X-API-Key") or "X-API-Key",
        "api_key": _get_config_value("RAG_API_KEY", ""),
        "verify_tls": _get_config_value("RAG_VERIFY_TLS", "true").lower() != "false",
        "show_technical": False,
        "last_document_id": "",
        "last_ingest": None,
        "last_search": None,
        "last_ask": None,
        "debug_ui": False,
        "documents": [],
        "last_http": None,
    }

    for key, default in defaults.items():
        st.session_state.setdefault(key, default)

    # Normalize even if user edited it in the UI or a previous session value exists
    st.session_state.api_base_url = _normalize_api_base_url(str(st.session_state.api_base_url))


def _http_headers() -> dict[str, str]:
    """Build request headers including optional API key."""
    headers: dict[str, str] = {"Accept": "application/json"}
    api_key = (st.session_state.api_key or "").strip()
    if api_key:
        headers[st.session_state.api_key_header] = api_key
    return headers


def _request_json(
    method: str,
    path: str,
    *,
    params: Optional[dict[str, Any]] = None,
    json_body: Optional[dict[str, Any]] = None,
    files: Optional[dict[str, Any]] = None,
) -> tuple[int, Any, dict[str, str]]:
    """Call the FastAPI backend and return (status_code, parsed_body, response_headers)."""
    base_url = _normalize_api_base_url(str(st.session_state.api_base_url))
    url = f"{base_url}{path}"
    headers = _http_headers()

    st.session_state.last_http = {
        "method": method.upper(),
        "url": url,
        "params": params,
        "json_keys": sorted(json_body.keys()) if isinstance(json_body, dict) else [],
        "has_files": bool(files),
    }

    if st.session_state.debug_ui:
        logger.info("HTTP %s %s params=%s json=%s files=%s", method, url, params, bool(json_body), bool(files))

    try:
        r = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            params=params,
            json=json_body,
            files=files,
            timeout=float(st.session_state.timeout_s),
            verify=bool(st.session_state.verify_tls),
        )
        elapsed = getattr(r, "elapsed", None)
        elapsed_s = None
        if elapsed is not None:
            try:
                elapsed_s = float(elapsed.total_seconds())
            except Exception:
                elapsed_s = None

        st.session_state.last_http = {
            **(st.session_state.last_http or {}),
            "status_code": r.status_code,
            "elapsed_s": elapsed_s,
            "content_type": r.headers.get("content-type", ""),
        }
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to reach API at {base_url}: {e}") from e

    content_type = r.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        try:
            return r.status_code, r.json(), dict(r.headers)
        except Exception:
            return r.status_code, r.text, dict(r.headers)

    return r.status_code, r.text, dict(r.headers)


def _render_api_error(payload: Any, status_code: int, response_headers: dict[str, str]) -> None:
    """Render backend errors with user-friendly messaging and optional technical details."""
    retry_after = response_headers.get("Retry-After") or response_headers.get("retry-after")

    message = "Request failed"
    details: Any = None
    request_id: Optional[str] = None

    if isinstance(payload, dict):
        message = str(payload.get("error") or message)
        details = payload.get("details")
        request_id = payload.get("request_id")
    else:
        message = str(payload)

    st.error(f"Error ({status_code}): {message}")

    if retry_after:
        st.info(f"Retry-After: {retry_after} seconds")

    if st.session_state.get("show_technical") and (details is not None or request_id is not None):
        with st.expander("Technical details", expanded=False):
            if request_id:
                st.caption(f"request_id: {request_id}")
            if details is not None:
                st.json(details)
            else:
                st.write("No additional details.")


def _page_health() -> None:
    """Health page: verify backend + Qdrant connectivity as exposed by the API."""
    st.subheader("Health & Diagnostics", help="Checks if the FastAPI service and Qdrant are reachable.")
    cols = st.columns([2, 1], vertical_alignment="top")

    with cols[0]:
        ok, data = _cached_health(st.session_state.api_base_url, _http_headers(), float(st.session_state.timeout_s))
        if ok:
            st.success("API is reachable")
            st.json(data)
        else:
            st.error("API is not reachable")
            st.code(str(data))

    with cols[1]:
        st.markdown("**Quick checks**")
        st.caption("Run lightweight checks without sending documents.")
        if st.button("Check /health/live", use_container_width=True, help="Calls GET /health/live"):
            code, payload, hdrs = _request_json("GET", "/health/live")
            if 200 <= code < 300:
                st.success("Live OK")
                st.json(payload)
            else:
                _render_api_error(payload, code, hdrs)

        if st.button("Check /health/ready", use_container_width=True, help="Calls GET /health/ready"):
            code, payload, hdrs = _request_json("GET", "/health/ready")
            if 200 <= code < 300:
                st.success("Ready OK")
                st.json(payload)
            else:
                _render_api_error(payload, code, hdrs)


def _remember_ingest(payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    document_id = str(payload.get("document_id") or "").strip()
    if not document_id:
        return
    filename = str(payload.get("filename") or "").strip()
    record = {
        "document_id": document_id,
        "filename": filename,
        "chunks": payload.get("chunks"),
        "pages": payload.get("pages"),
    }

    docs: list[dict[str, Any]] = list(st.session_state.get("documents") or [])
    for i, existing in enumerate(docs):
        if str(existing.get("document_id") or "") == document_id:
            docs[i] = {**existing, **record}
            break
    else:
        docs.insert(0, record)

    st.session_state.documents = docs
    st.session_state.last_document_id = document_id


def _page_ingest() -> None:
    """Ingestion page: upload a PDF for chunking/embedding/storage."""
    st.subheader(
        "Ingest PDF", help="Uploads a PDF to the backend to extract text, chunk it, embed it, and store it in Qdrant."
    )
    st.caption("Endpoint: POST /ingest")

    c1, c2 = st.columns([2, 1], vertical_alignment="top")

    with c1:
        uploaded = st.file_uploader(
            "PDF file",
            type=["pdf"],
            accept_multiple_files=False,
            help="Select a PDF to ingest. The backend enforces a max upload size; large PDFs may be rejected.",
            key="demo_ingest_file",
        )

        ingest_clicked = st.button(
            "Ingest",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None,
            help="Uploads the selected PDF to POST /ingest.",
            key="demo_ingest_button",
        )

        if ingest_clicked and uploaded is not None:
            with st.spinner("Uploading and processing..."):
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                code, payload, hdrs = _request_json("POST", "/ingest", files=files)

            if 200 <= code < 300:
                st.success("Ingestion completed")
                st.session_state.last_ingest = payload
                _remember_ingest(payload)
                st.write(f"Document ID: {st.session_state.last_document_id}")
                if st.session_state.get("show_technical"):
                    with st.expander("Technical response", expanded=False):
                        st.json(payload)
            else:
                _render_api_error(payload, code, hdrs)

    with c2:
        st.markdown("**Session**")
        st.caption("Convenience values stored in the browser session.")
        st.text_input(
            "Last document_id",
            value=st.session_state.last_document_id,
            key="last_document_id",
            help="Auto-populated after a successful ingest. You can overwrite it to filter search/ask.",
        )
        if st.session_state.last_ingest and st.session_state.get("show_technical"):
            with st.expander("Technical: last ingest response", expanded=False):
                st.json(st.session_state.last_ingest)

        with st.expander("Delete ingested document", expanded=False):
            st.caption("Deletes all chunks for a document_id from the vector store (Qdrant).")
            delete_document_id = st.text_input(
                "document_id to delete",
                value=st.session_state.last_document_id,
                key="delete_document_id",
                help="Document ID returned by ingest. This operation removes the document's chunks from Qdrant.",
            )
            confirm_delete = st.checkbox(
                "I understand this cannot be undone",
                value=False,
                key="confirm_delete",
                help="Required to enable deletion.",
            )
            do_delete = st.button(
                "Delete",
                use_container_width=True,
                disabled=not (confirm_delete and delete_document_id.strip()),
                help="Calls DELETE /ingest/{document_id}.",
            )

            if do_delete:
                with st.spinner("Deleting from vector store..."):
                    code, payload, hdrs = _request_json(
                        "DELETE",
                        f"/ingest/{delete_document_id.strip()}",
                    )

                if 200 <= code < 300:  # noqa: PLR2004
                    st.success("Deleted")
                    if st.session_state.last_document_id == delete_document_id.strip():
                        st.session_state.last_document_id = ""
                    if st.session_state.get("show_technical"):
                        with st.expander("Technical response", expanded=False):
                            st.json(payload)
                else:
                    _render_api_error(payload, code, hdrs)


def _results_to_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert search results into a DataFrame suitable for display."""
    rows: list[dict[str, Any]] = []
    for item in results:
        metadata = item.get("metadata") or {}
        rows.append(
            {
                "score": item.get("score"),
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "page_number": item.get("page_number"),
                "text_preview": (item.get("text") or "")[:200],
                "metadata": metadata,
            }
        )
    return pd.DataFrame(rows)


def _run_search(query: str, top_k: int, score_threshold: float, document_id: str, transport: str) -> None:
    with st.spinner("Searching..."):
        if transport == "GET":
            params: dict[str, Any] = {
                "q": query,
                "top_k": int(top_k),
                "score_threshold": float(score_threshold),
            }
            if document_id.strip():
                params["document_id"] = document_id.strip()
            code, payload, hdrs = _request_json("GET", "/query/search", params=params)
        else:
            body: dict[str, Any] = {
                "query": query,
                "top_k": int(top_k),
                "score_threshold": float(score_threshold),
                "document_id": document_id.strip() or None,
            }
            code, payload, hdrs = _request_json("POST", "/query/search", json_body=body)

    if 200 <= code < 300:
        st.session_state.last_search = payload
        st.success("Search completed")
    else:
        _render_api_error(payload, code, hdrs)
        st.session_state.last_search = None


def _render_search_results() -> None:
    payload = st.session_state.last_search
    if not payload:
        st.info("Run a search to see results.", icon="ℹ️")
        return

    if not isinstance(payload, dict):
        st.error("Unexpected response format from /query/search")
        st.write(payload)
        return

    results = payload.get("results") or []
    total = payload.get("total")
    st.markdown(f"**Results: {total}**")
    st.caption("Total number of results returned by the API.")

    if not results:
        st.warning("No results found.")
        return

    df = _results_to_table(results)

    st.dataframe(
        df.drop(columns=["metadata"]),
        use_container_width=True,
        height=320,
    )

    chart_df = df[["chunk_id", "score"]].copy()
    chart_df["chunk_id"] = chart_df["chunk_id"].astype(str)

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("score:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("chunk_id:N", sort="-x"),
            tooltip=["chunk_id", "score"],
        )
        .properties(height=min(500, 28 * len(chart_df)))
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("**Details**")
    st.caption("Inspect full chunk text and metadata.")
    for idx, item in enumerate(results, 1):
        title = f"{idx}. score={item.get('score')} page={item.get('page_number')} chunk_id={item.get('chunk_id')}"
        with st.expander(title, expanded=False):
            st.text(item.get("text") or "")
            if st.session_state.get("show_technical"):
                with st.expander("Technical metadata", expanded=False):
                    st.json(item.get("metadata") or {})


def _page_search() -> None:  # noqa: PLR0915
    """Search page: semantic retrieval over ingested chunks."""
    st.subheader("Semantic Search", help="Retrieves the most relevant chunks based on your query.")
    st.caption("Endpoint: POST /query/search (or GET /query/search)")

    left, right = st.columns([1, 1], vertical_alignment="top")

    with left:
        query = st.text_area(
            "Query",
            value="",
            height=110,
            help="Natural-language query. This is embedded and used for vector similarity search.",
        )
        top_k = int(
            st.number_input(
                "top_k",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="Maximum number of chunks to return.",
            )
        )
        score_threshold = float(
            st.slider(
                "score_threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                help="Minimum similarity score. Increase to filter out weaker matches.",
            )
        )
        document_id = st.text_input(
            "document_id (optional)",
            value=st.session_state.last_document_id,
            help="If set, restricts search results to a single ingested document.",
        )
        transport = st.radio(
            "Request type",
            options=["POST", "GET"],
            horizontal=True,
            help="Choose POST for larger queries; GET is a convenience endpoint.",
        )

        run = st.button(
            "Search",
            type="primary",
            use_container_width=True,
            disabled=not query.strip(),
            help="Runs semantic search against Qdrant and returns the top results.",
        )

        if run:
            _run_search(query, top_k, score_threshold, document_id, transport)

    with right:
        _render_search_results()


def _page_ask() -> None:
    """Ask page: retrieve context and generate a grounded answer."""
    st.subheader("Ask (RAG)", help="Retrieves relevant chunks then asks Gemini to answer using only that context.")
    st.caption("Endpoint: POST /query/ask")

    left, right = st.columns([1, 1], vertical_alignment="top")

    with left:
        question = st.text_area(
            "Question",
            value="",
            height=110,
            help="Your question. The backend retrieves relevant chunks then calls Gemini to generate an answer.",
        )
        top_k = st.number_input(
            "top_k",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Number of chunks to retrieve for grounding before generating an answer.",
        )
        document_id = st.text_input(
            "document_id (optional)",
            value=st.session_state.last_document_id,
            help="If set, retrieval is limited to that document before answering.",
        )
        temperature = st.slider(
            "temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Higher values make the model more creative; lower values are more deterministic.",
        )

        run = st.button(
            "Ask",
            type="primary",
            use_container_width=True,
            disabled=not question.strip(),
            help="Runs retrieval + generation. Requires a valid Gemini API key configured on the backend.",
        )

        if run:
            body: dict[str, Any] = {
                "question": question,
                "top_k": int(top_k),
                "document_id": document_id.strip() or None,
                "temperature": float(temperature),
            }
            with st.spinner("Retrieving context and generating answer..."):
                code, payload, hdrs = _request_json("POST", "/query/ask", json_body=body)

            if 200 <= code < 300:
                st.session_state.last_ask = payload
                st.success("Answer generated")
            else:
                _render_api_error(payload, code, hdrs)
                st.session_state.last_ask = None

    with right:
        payload = st.session_state.last_ask
        if not payload:
            st.info("Ask a question to see the answer and citations.", icon="ℹ️")
            return

        if not isinstance(payload, dict):
            st.error("Unexpected response format from /query/ask")
            st.write(payload)
            return

        st.markdown("### Answer")
        st.write(payload.get("answer") or "")

        st.markdown("### Sources")
        sources = payload.get("sources") or []
        if not sources:
            st.warning("No citations returned.")
            return

        for src in sources:
            source_id = src.get("source_id")
            doc_id = src.get("document_id")
            page = src.get("page_number")
            score = src.get("relevance_score")
            title = f"Source {source_id} · doc={doc_id} · page={page} · score={score}"
            with st.expander(title, expanded=False):
                st.write(src.get("text_preview") or "")
                if st.session_state.get("show_technical"):
                    with st.expander("Technical source", expanded=False):
                        st.json(src)


def _user_doc_options() -> tuple[list[dict[str, Any]], list[tuple[str, str]], dict[str, str]]:
    docs: list[dict[str, Any]] = list(st.session_state.get("documents") or [])
    options: list[tuple[str, str]] = []
    for d in docs:
        doc_id = str(d.get("document_id") or "").strip()
        if not doc_id:
            continue
        name = str(d.get("filename") or "").strip() or "Document"
        short_id = doc_id[-6:] if len(doc_id) > 6 else doc_id
        options.append((f"{name} · {short_id}", doc_id))
    label_to_id: dict[str, str] = dict(options)
    return docs, options, label_to_id


def _user_ingest() -> None:
    st.markdown("**Ingest PDF**")
    uploaded = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        accept_multiple_files=False,
        key="user_ingest_file",
    )
    ingest_clicked = st.button(
        "Ingest",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None,
        key="user_ingest_button",
    )

    if ingest_clicked and uploaded is not None:
        with st.spinner("Uploading and processing..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
            code, payload, hdrs = _request_json("POST", "/ingest", files=files)

        if 200 <= code < 300:
            st.success("Ingested")
            st.session_state.last_ingest = payload
            _remember_ingest(payload)
            st.rerun()
        else:
            _render_api_error(payload, code, hdrs)


def _user_delete(docs: list[dict[str, Any]], options: list[tuple[str, str]], label_to_id: dict[str, str]) -> None:
    st.markdown("**Delete a document**")
    if not options:
        st.info("No documents in this session yet. Ingest a PDF to enable deletion.", icon="ℹ️")
        return

    selected_label = st.selectbox(
        "Choose a document",
        options=[label for label, _ in options],
        key="user_delete_select",
    )
    confirm_delete = st.checkbox(
        "I understand this cannot be undone",
        value=False,
        key="user_confirm_delete",
    )
    do_delete = st.button(
        "Delete",
        use_container_width=True,
        disabled=not (confirm_delete and selected_label),
        key="user_delete_button",
    )

    if not do_delete:
        return

    delete_document_id = label_to_id.get(selected_label, "")
    with st.spinner("Deleting..."):
        code, payload, hdrs = _request_json("DELETE", f"/ingest/{delete_document_id}")

    if 200 <= code < 300:  # noqa: PLR2004
        st.success("Deleted")
        st.session_state.documents = [d for d in docs if str(d.get("document_id") or "") != delete_document_id]
        if st.session_state.last_document_id == delete_document_id:
            st.session_state.last_document_id = ""
        st.rerun()
    else:
        _render_api_error(payload, code, hdrs)


def _user_ask(options: list[tuple[str, str]], label_to_id: dict[str, str]) -> None:
    st.markdown("**Ask a question**")
    question = st.text_area(
        "Your question",
        value="",
        height=140,
        key="user_question",
    )
    top_k = int(
        st.number_input(
            "Results to use",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="user_top_k",
        )
    )
    temperature = float(
        st.slider(
            "Answer creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            key="user_temperature",
        )
    )

    scope = st.radio(
        "Knowledge base",
        options=["Files at hand", "General knowledge"],
        horizontal=True,
        key="user_scope_mode",
        index=0 if options else 1,
        help="Files at hand limits answers to PDFs you uploaded in this session (NotebookLM-style).",
    )

    document_ids: Optional[list[str]] = None
    if scope == "Files at hand":
        if not options:
            st.info("Upload a PDF first to use Files at hand.", icon="ℹ️")
        else:
            labels = [label for label, _ in options]
            selected_labels = st.multiselect(
                "Use these documents",
                options=labels,
                default=labels,
                key="user_scope_docs",
            )
            document_ids = [label_to_id[lbl] for lbl in selected_labels if lbl in label_to_id]
            if not document_ids:
                st.warning("Select at least one document, or switch to General knowledge.")

    ask_disabled = (not question.strip()) or (scope == "Files at hand" and not (document_ids or []))

    run = st.button(
        "Ask",
        type="primary",
        use_container_width=True,
        disabled=ask_disabled,
        key="user_ask_button",
    )

    if run:
        body: dict[str, Any] = {
            "question": question,
            "top_k": top_k,
            "document_id": None,
            "document_ids": document_ids if scope == "Files at hand" else None,
            "temperature": temperature,
        }
        with st.spinner("Answering..."):
            code, payload, hdrs = _request_json("POST", "/query/ask", json_body=body)

        if 200 <= code < 300:
            st.session_state.last_ask = payload
        else:
            _render_api_error(payload, code, hdrs)
            st.session_state.last_ask = None

    payload = st.session_state.last_ask
    if payload and isinstance(payload, dict):
        st.markdown("### Answer")
        st.write(payload.get("answer") or "")

        sources = payload.get("sources") or []
        if sources:
            with st.expander("Sources", expanded=False):
                for src in sources:
                    page = src.get("page_number")
                    title = f"Page {page}"
                    with st.expander(title, expanded=False):
                        st.write(src.get("text_preview") or "")


def _page_user() -> None:  # noqa: C901, PLR0912, PLR0915
    st.subheader("Ask, ingest, and delete")

    left, right = st.columns([1, 1], vertical_alignment="top")

    docs, options, label_to_id = _user_doc_options()

    with left:
        _user_ingest()
        _user_delete(docs, options, label_to_id)

    with right:
        _user_ask(options, label_to_id)


def main() -> None:
    """Streamlit entrypoint."""
    st.set_page_config(
        page_title="Enterprise RAG",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
          #MainMenu {visibility: hidden;}
          footer {visibility: hidden;}
          header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    _init_session_state()
    _require_auth()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    st.title("Enterprise RAG")

    tab_demo, tab_user = st.tabs(["Leads / Demo", "User"])

    with tab_user:
        _page_user()

    with tab_demo:
        st.subheader("Behind the scenes")
        st.caption("API health, configuration, and full workflows for demos/interviews.")

        with st.expander("Settings", expanded=False):
            st.text_input("API base URL", key="api_base_url")
            st.text_input("API key header", key="api_key_header")
            st.text_input("API key", type="password", key="api_key")
            st.number_input(
                "Timeout (seconds)",
                min_value=1,
                max_value=300,
                value=int(st.session_state.timeout_s),
                key="timeout_s",
            )
            st.checkbox("Verify TLS", key="verify_tls")
            st.checkbox("Show technical details", key="show_technical")
            st.checkbox("Debug UI", key="debug_ui")

        if st.session_state.get("authenticated"):
            if st.button("Sign out", use_container_width=True, key="demo_sign_out"):
                st.session_state.authenticated = False
                st.rerun()

        if st.session_state.get("show_technical") and st.session_state.get("last_http"):
            with st.expander("Last API call", expanded=False):
                st.json(st.session_state.last_http)

        with st.expander("Health & diagnostics", expanded=False):
            _page_health()

        with st.expander("Ingest (advanced)", expanded=False):
            _page_ingest()

        with st.expander("Search (advanced)", expanded=False):
            _page_search()

        with st.expander("Ask (advanced)", expanded=False):
            _page_ask()


if __name__ == "__main__":
    main()
