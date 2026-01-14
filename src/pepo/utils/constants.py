"""Constants used across the PEPO codebase."""

SMOLLM_CHAT_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- messages[0]['content'] }}"
    "{%- else %}"
    "{{- 'You are a helpful AI assistant.' }}"
    "{%- endif %}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'user' %}"
    "{{- '\n\n### User:\n' + message['content'] }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- '\n\n### Assistant:\n' + message['content'] }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '\n\n### Assistant:\n' }}"
    "{%- endif %}"
)
