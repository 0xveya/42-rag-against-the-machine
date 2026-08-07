"""Allow ``python -m rag_against_the_machine``."""

from .main import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nServer stopped. Goodbye!")
