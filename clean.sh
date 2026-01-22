BASE=/egr/research-sprintai/baliahsa/projects/DeepfakeBench/logs/training
DEST=$BASE/incompleted

mkdir -p "$DEST"

for d in "$BASE"/*; do
    if [ -d "$d" ] && [ "$(basename "$d")" != "incompleted" ]; then
        if [ ! -d "$d/test" ]; then
            mv "$d" "$DEST/"
        fi
    fi
done