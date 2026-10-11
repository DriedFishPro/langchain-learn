import langchain_openai.chat_models.base as openai_base

# ==========================================
# 修复 LangChain 丢失底层扩展字段的问题
# ==========================================
def apply_patches():
    # --- 修复 Stream 流式调用 ---
    _original_stream_convert = openai_base._convert_delta_to_message_chunk

    def _patched_stream_convert(_dict, default_class):
        # 【1. 埋点：打印 LangChain 丢给转换器的原始入参】，只要控制台打出了这行，就说明切点 100% 命中了！
        # print(f"【底层拦截入参】: {_dict}")

        # 2. 执行 LangChain 原逻辑
        message_chunk = _original_stream_convert(_dict, default_class)

        try:
            reasoning = _dict.get("reasoning_content")
            if reasoning:
                message_chunk.additional_kwargs["reasoning_content"] = reasoning
                # print(f"\n【切面注入成功】: {reasoning}")
        except Exception as e:
            pass  # 防御性编程
        return message_chunk
    # 替换底层框架的转换器（全局生效）
    openai_base._convert_delta_to_message_chunk = _patched_stream_convert

    # --- 修复 Invoke 非流式调用 ---
    _original_invoke_convert = openai_base._convert_dict_to_message

    def _patched_invoke_convert(_dict):
        message = _original_invoke_convert(_dict)
        try:
            reasoning = _dict.get("reasoning_content")
            if reasoning:
                message.additional_kwargs["reasoning_content"] = reasoning
        except Exception:
            pass
        return message

    openai_base._convert_dict_to_message = _patched_invoke_convert
    print("模型思考过程修复补丁已挂载")