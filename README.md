# pruebaIA_Solvex
Prueba tecnica de Ingeniero de IA para Solvex

### Pasos para usar el bot
1. Usar python -m uvicorn main:app --reload para prender el servidor
2. Usar el el link que te proporciona uvicorn con /docs al final (algo asi: http://127.0.0.1:8000/docs)
3. Al estar en la pagina solo tienes que clikear POST /query
4. Cargará la página correcta y ahí pondrás lo que toca en cada cosa (user_id: "usuario_ejemplo" y query: "mensaje de pregunta")

### Pasos para correr la calidad del código con pytest
1. Correr el siguiente código en tu terminal: pytest tests/test_pipeline.py -v

### Cosas a tener en cuenta
1. Para correr todo esto debes tener un PC con por lo menos 8 de VRAM sino el modelo tardará mucho o crasheará, sino en la parte de limitación de los bits eso puede ser cambiado para uso o hacer un cambio del modelo
2. En pipeline.py debes tener en cuenta usar tu token de HuggingFace con permiso del modelo, sino prueba uno con el que si tengas permiso que sea para el mismo uso, sino, no funcionará.

### Comentarios
Lo del Dockerfile está ahí pero no supe como usarlo.
