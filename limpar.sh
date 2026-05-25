echo ""
#!/bin/bash
set -e

# --- LIMPEZA ---
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Limpando artefatos anteriores..."

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
rm -rf data/faiss_index
rm -f  fine_tuning/data/train.jsonl fine_tuning/data/valid.jsonl
rm -f  fine_tuning/adapters/0000*_adapters.safetensors \
       fine_tuning/adapters/adapters.safetensors
rm -rf fine_tuning/fused_model
python3 -c "
import sqlite3, os
db = os.path.join('data', 'prontuarios.db')
if os.path.exists(db):
    conn = sqlite3.connect(db)
    conn.execute('DELETE FROM atendimentos')
    conn.execute('DELETE FROM sqlite_sequence WHERE name=\"atendimentos\"')
    conn.commit()
    conn.close()
"
echo "[Limpeza] FAISS index, train/valid JSONL, adapters, fused_model e audit DB removidos."
echo ""