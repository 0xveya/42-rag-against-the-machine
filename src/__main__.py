"""Allow ``python -m src`` to run the project CLI."""

from src.rag_against_the_machine.main import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nServer stopped. Goodbye!")
