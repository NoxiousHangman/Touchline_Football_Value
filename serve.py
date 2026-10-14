"""Production entry point: one process, four HTTP threads, one shared updater."""
import atexit
import logging
import os


def main():
    from waitress import serve
    from app import Predictor
    from src.web_server import WebApplication

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    predictor = Predictor()
    application = WebApplication(predictor, public=True,
        public_url=os.environ.get("PUBLIC_BASE_URL", os.environ.get("RENDER_EXTERNAL_URL", "")))
    if os.environ.get("TOUCHLINE_AUTO_UPDATE", "1") == "1":
        predictor.live.start_updater()
    atexit.register(predictor.live.stop_updater)
    serve(application, host=os.environ.get("HOST", "127.0.0.1"),
          port=int(os.environ.get("PORT", "8765")), threads=4,
          connection_limit=100, channel_timeout=30,
          max_request_header_size=16384, max_request_body_size=1024,
          expose_tracebacks=False, ident="Touchline")


if __name__ == "__main__":
    main()
