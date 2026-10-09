"""Offline SSE integration check, run with the runner image's OpenHands venv.

This does not call a model or read benchmark cases. It exercises the actual
OpenHands -> LiteLLM -> HTTP SSE -> assembled response -> usage ledger path.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from harness.backends.openhands.driver import _install_chat_streaming_compat


def main():
    from openhands.core.config import LLMConfig
    from openhands.llm.llm import LLM

    requests = []
    provider_usage = {
        "prompt_tokens": 42, "completion_tokens": 12, "total_tokens": 54,
        "prompt_tokens_details": {"cached_tokens": 20},
        "completion_tokens_details": {"reasoning_tokens": 8},
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            deltas = [
                {"role": "assistant", "reasoning_content": "check ", "content": ""},
                {"reasoning_content": "tools"},
                {"content": "Running tools."},
                {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                 "function": {"name": "execute_bash", "arguments": '{"command":'}}]},
                {"tool_calls": [{"index": 1, "id": "call_2", "type": "function",
                                 "function": {"name": "execute_bash", "arguments": '{"command":'}}]},
                {"tool_calls": [{"index": 0, "function": {"arguments": '"true"}'}}]},
                {"tool_calls": [{"index": 1, "function": {"arguments": '"pwd"}'}}]},
                {},
            ]
            base = {"id": "chatcmpl-stream-smoke", "object": "chat.completion.chunk",
                    "created": 1789500000, "model": "qwen3.8-max"}
            for i, delta in enumerate(deltas):
                chunk = {**base, "choices": [{"index": 0, "delta": delta,
                         "finish_reason": "tool_calls" if i == len(deltas) - 1 else None}]}
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(("data: " + json.dumps({**base, "choices": [], "usage": provider_usage})
                              + "\n\ndata: [DONE]\n\n").encode())
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    stats = {}
    restore = _install_chat_streaming_compat(stats)
    try:
        llm = LLM(LLMConfig(
            model="openai/qwen3.8-max", api_key="offline-test-key",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            max_input_tokens=1000000, max_output_tokens=131072,
            native_tool_calling=True, timeout=30, num_retries=1,
        ), service_id="stream-smoke")
        response = llm.completion(
            messages=[{"role": "user", "content": "Offline test: call both tools."}],
            tools=[{"type": "function", "function": {"name": "execute_bash",
                    "parameters": {"type": "object", "properties": {
                        "command": {"type": "string"}}, "required": ["command"]}}}],
        )
        assert len(requests) == 1 and requests[0]["stream"] is True
        assert requests[0]["stream_options"]["include_usage"] is True
        assert requests[0].get("max_completion_tokens", requests[0].get("max_tokens")) == 131072
        message = response.choices[0].message
        assert message.content == "Running tools."
        assert message.reasoning_content == "check tools"
        assert [json.loads(call.function.arguments) for call in message.tool_calls] == [
            {"command": "true"}, {"command": "pwd"}]
        assert response.usage.prompt_tokens == 42 and response.usage.completion_tokens == 12
        assert response.usage.prompt_tokens_details.cached_tokens == 20
        assert response.usage.completion_tokens_details.reasoning_tokens == 8
        assert len(llm.metrics.token_usages) == 1
        assert llm.metrics.token_usages[0].prompt_tokens == 42
        assert llm.metrics.token_usages[0].completion_tokens == 12
        assert llm.metrics.token_usages[0].cache_read_tokens == 20
        assert stats["completed"] == 1 and stats["failed"] == 0
        print("STREAM_SMOKE_PASS: SSE, interleaved tools, reasoning, provider usage, OpenHands metrics")
    finally:
        restore()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
