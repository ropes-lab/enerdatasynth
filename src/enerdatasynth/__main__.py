"""`python3 -m enerdatasynth <command>`: same as the `enerdatasynth` command,
but it doesn't depend on pip's script folder being on your PATH."""
from enerdatasynth.cli import main

if __name__ == "__main__":
    main()
