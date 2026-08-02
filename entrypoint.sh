#!/bin/sh
# entrypoint.sh
# Genera /MoneyPrinterTurbo/config.toml a partir de variables de entorno
# antes de arrancar la app. Railway no soporta "docker run -v archivo",
# así que este script reemplaza ese paso.

set -e

CONFIG_PATH="/MoneyPrinterTurbo/config.toml"

OPENAI_MODEL_NAME="${OPENAI_MODEL_NAME:-gpt-4o-mini}"

echo "Generando config.toml desde variables de entorno..."

cat > "$CONFIG_PATH" <<EOF
log_level = "INFO"
listen_host = "0.0.0.0"
listen_port = 8080

[app]

hide_config = false
edge_tts_timeout = 30
tls_verify = true

video_source = "pexels"
pexels_api_keys = ["${PEXELS_API_KEY}"]
pixabay_api_keys = []
coverr_api_keys = []
twelvelabs_api_keys = []
twelvelabs_rerank_terms = false
match_materials_to_script = false

sonilo_api_key = ""
sonilo_base_url = "https://api.sonilo.com"
sonilo_timeout = 600

llm_provider = "openai"

moonshot_api_key = ""
moonshot_base_url = ""
moonshot_model_name = ""

openai_api_key = "${OPENAI_API_KEY}"
openai_base_url = ""
openai_model_name = "${OPENAI_MODEL_NAME}"

gemini_api_key = ""
gemini_model_name = ""

deepseek_api_key = ""
deepseek_base_url = ""
deepseek_model_name = ""

qwen_api_key = ""
qwen_model_name = ""

azure_api_key = ""
azure_base_url = ""
azure_model_name = ""
azure_api_version = "2024-02-15-preview"

volcengine_api_key = ""
volcengine_base_url = ""
volcengine_model_name = ""

grok_api_key = ""
grok_base_url = ""
grok_model_name = ""

minimax_api_key = ""
minimax_base_url = ""
minimax_model_name = ""

mimo_api_key = ""
mimo_base_url = ""
mimo_model_name = ""

cloudflare_api_key = ""
cloudflare_account_id = ""
cloudflare_gateway_id = ""
cloudflare_model_name = ""

modelscope_api_key = ""
modelscope_base_url = ""
modelscope_model_name = ""

aihubmix_api_key = ""
aihubmix_base_url = ""
aihubmix_model_name = ""

aimlapi_api_key = ""
aimlapi_base_url = ""
aimlapi_model_name = ""

evolink_api_key = ""
evolink_base_url = ""
evolink_model_name = ""

ollama_base_url = ""
ollama_model_name = ""

oneapi_api_key = ""
oneapi_base_url = ""
oneapi_model_name = ""

litellm_model_name = ""

groq_api_key = ""
groq_base_url = ""
groq_model_name = ""

pollinations_api_key = ""
pollinations_base_url = ""
pollinations_model_name = ""

mimo_tts_model_name = "mimo-v2.5-tts"
mimo_tts_style_prompt = "请用自然、清晰、适合短视频旁白的语气朗读。"

subtitle_provider = "edge"

endpoint = ""
material_directory = ""

enable_redis = false
redis_host = "localhost"
redis_port = 6379
redis_db = 0
redis_password = ""

max_concurrent_tasks = 5
max_queued_tasks = 100

upload_post_enabled = false
upload_post_api_key = ""
upload_post_username = ""
upload_post_platforms = ["tiktok", "instagram"]
upload_post_auto_upload = false
upload_post_youtube_privacy_status = "public"
upload_post_max_pending_tasks = 10

[whisper]
model_size = "large-v3"
device = "cpu"
compute_type = "int8"

[proxy]

[azure]
speech_key = "${AZURE_SPEECH_KEY}"
speech_region = "${AZURE_SPEECH_REGION}"

[siliconflow]
api_key = ""

[elevenlabs]
api_key = ""
model_id = "eleven_multilingual_v2"
music_model_id = "music_v2"
music_timeout = 600

[chatterbox]
base_url = "http://127.0.0.1:4123/v1"
api_key = ""
model_id = "chatterbox"
voices = ["default-Female"]

[ui]
hide_log = false
EOF

echo "config.toml generado correctamente."

exec "$@"
