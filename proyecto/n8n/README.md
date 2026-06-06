# n8n - Bridge WhatsApp del agente Sandwich Qbano (Taller 3, Fase 6)

Este modulo conecta WhatsApp (via Twilio Sandbox) con la API REST del agente
(`api/main.py`) usando n8n como orquestador low-code. Cumple el Punto 3 de la
Ruta A del Taller 3.

## Arquitectura

```
Usuario WhatsApp
      |
      v
Twilio Sandbox (+1 415 523 8886)
      |
      | webhook HTTPS
      v
ngrok (https://<id>.ngrok-free.dev) ---> n8n local (puerto 5678)
                                              |
                                              v
                                  Workflow "Sandwich Qbano - WhatsApp Bridge"
                                              |
                                  Webhook -> Extraer campos -> HTTP Request -> HTTP Twilio
                                              |
                                              v
                              FastAPI local (http://host.docker.internal:8000/chat)
                                              |
                                              v
                              run_agent() -> PostgresSaver + tools Pydantic + Chroma
```

## Setup local (one-shot)

Todo corre con Docker via `docker-compose.yml` en la raiz del proyecto. Servicios:

- `qbano_postgres`: BD del agente (puerto 5432).
- `qbano_n8n`: n8n local (puerto 5678) con volumen persistente.

```bash
cd proyecto
docker compose up -d
```

## Workflow

El workflow esta versionado en `workflows/whatsapp_qbano.json`. Tiene 5 nodos:

1. **Webhook Twilio (entrante)** — POST `/webhook/twilio-qbano` (form-urlencoded de Twilio).
2. **Extraer campos Twilio** — extrae `From`, `To`, `Body`; genera `thread_id` = numero sin `whatsapp:`.
3. **Llamar agente Qbano (FastAPI)** — POST `http://host.docker.internal:8000/chat`, usando `provider=opencode_go` y `model=kimi-k2.6` para que WhatsApp ejecute el LLM en la nube de OpenCode Go y no en Ollama local.
4. **Responder por WhatsApp (Twilio API)** — POST `https://api.twilio.com/2010-04-01/Accounts/{SID}/Messages.json` con Basic Auth (credencial `Twilio Qbano`).
5. **ACK 200 a Twilio** — devuelve `<Response></Response>` para cerrar el ciclo HTTP.

### Importar el workflow (si se borra)

n8n CLI desde el container (con n8n detenido):

```bash
docker stop qbano_n8n
docker run --rm \
  -v proyecto_n8n_data:/home/node/.n8n \
  -v $PWD/n8n/workflows:/wf \
  --entrypoint sh \
  docker.n8n.io/n8nio/n8n:latest \
  -c "n8n import:workflow --input=/wf/whatsapp_qbano.json --userId=<USER_ID>"
docker start qbano_n8n
```

### Credencial Twilio

Crear en n8n -> Credentials -> "Twilio API":

- Account SID: el del dashboard de Twilio.
- Auth Token: el del dashboard de Twilio.
- Nombre: `Twilio Qbano`.

El nodo "Responder por WhatsApp" referencia esta credencial por nombre.

## Exposicion publica con ngrok

Twilio Sandbox necesita una URL HTTPS publica para entregar el webhook.

```bash
ngrok config add-authtoken <YOUR_TOKEN>   # solo la primera vez
ngrok http 5678                            # tunel hacia n8n local
```

ngrok imprime una URL del tipo `https://<random>.ngrok-free.dev`. La URL completa
del webhook que se debe pegar en Twilio es:

```
https://<random>.ngrok-free.dev/webhook/twilio-qbano
```

## Configurar Twilio Sandbox (UI, ~30s)

1. Login en https://console.twilio.com/
2. Menu: **Develop > Messaging > Try it out > Send a WhatsApp message**.
3. Pestana **Sandbox settings**.
4. Campo **WHEN A MESSAGE COMES IN**:
   - Metodo: `POST`.
   - URL: la del ngrok mas `/webhook/twilio-qbano`.
5. **Save**.

> El sandbox de Twilio no expone API publica para esta configuracion: hay que
> hacerlo desde el panel web. Es la unica accion manual del Taller 3.

## Verificacion

### Salud de todos los componentes

```bash
# FastAPI
curl http://127.0.0.1:8000/health   # debe responder {"status":"ok",...}

# n8n
curl http://localhost:5678/healthz   # debe responder {"status":"ok"}

# ngrok hacia n8n
curl http://127.0.0.1:4040/api/tunnels | jq '.tunnels[].public_url'

# Postgres
docker exec qbano_postgres pg_isready -U qbano
```

### Smoke test del webhook desde curl

```bash
curl -X POST http://localhost:5678/webhook/twilio-qbano \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "From=whatsapp:+573001234567" \
  --data-urlencode "To=whatsapp:+14155238886" \
  --data-urlencode "Body=Cual es el WhatsApp?"
# Debe devolver <Response></Response>
```

Luego en n8n -> Executions debe aparecer una ejecucion con `status=success`.

### Test real end-to-end

1. Asegurate de que tu numero de WhatsApp esta unido al sandbox (mensaje
   `join <palabra-secreta>` al `+14155238886`).
2. Manda cualquier pregunta desde tu WhatsApp al `+14155238886`.
3. En ~5-10 segundos llega la respuesta del agente.

## Auditoria post-ejecucion

Toda la conversacion queda registrada en Postgres (`conversation_messages`):

```sql
SELECT role, route, context_mode, llm_provider, left(content, 80) AS preview, created_at
FROM conversation_messages
WHERE thread_id = '+573001234567'   -- tu numero E.164, conserva el '+' y sin prefijo 'whatsapp:'
ORDER BY id DESC LIMIT 20;
```

> El workflow extrae `From` (formato Twilio `whatsapp:+57...`) y solo remueve el
> prefijo `whatsapp:`, dejando el `+` E.164 intacto. Eso es lo que termina
> guardado en la tabla, no la versión sin signo.

## Notas de seguridad

- No commitear `taller3/infodata.md` (esta fuera del repo, pero conviene rotar
  Account SID/Auth Token despues de la sustentacion).
- ngrok free tier rota la URL cada vez que se reinicia el tunel; para una URL
  permanente se requiere plan pago o dominio propio.
- El N8N_ENCRYPTION_KEY se genera automaticamente la primera vez; queda en el
  volumen `proyecto_n8n_data`. Si se pierde, las credenciales encriptadas se
  pierden tambien.
