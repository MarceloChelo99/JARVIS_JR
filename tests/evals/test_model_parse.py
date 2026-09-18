from jarvis_jr.evals.model_lmstudio import parse_reply


def test_parse_reply_with_tool_calls():
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "write_file", "arguments": '{"path": "a.txt", "content": "hi"}'},
                        }
                    ],
                }
            }
        ]
    }
    reply = parse_reply(response)
    assert reply.content == ""
    assert reply.tool_calls[0].tool == "write_file"
    assert reply.tool_calls[0].args == {"path": "a.txt", "content": "hi"}
    assert reply.tool_calls[0].call_id == "call_1"
    assert reply.raw_message["role"] == "assistant"


def test_parse_reply_plain_answer_and_malformed_args():
    plain = {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
    assert parse_reply(plain).content == "done"
    assert parse_reply(plain).tool_calls == ()

    malformed = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c", "function": {"name": "read_file", "arguments": "{not json"}}
                    ],
                }
            }
        ]
    }
    reply = parse_reply(malformed)
    assert reply.tool_calls[0].args == {"_raw": "{not json"}
