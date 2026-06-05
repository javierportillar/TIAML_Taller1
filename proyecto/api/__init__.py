"""API REST del agente Sandwich Qbano (Taller 3 - Ruta A, Punto 2).

Expone el agente conversacional como servicio web HTTP para que N8N, Twilio o
cualquier otro cliente externo (incluyendo el futuro webhook de WhatsApp) pueda
conversar con el LLM sin depender de Streamlit. Reutiliza completamente las
capas internas del proyecto (run_agent, PostgresSaver, herramientas Pydantic,
selector multi-LLM, memoria persistente).
"""
