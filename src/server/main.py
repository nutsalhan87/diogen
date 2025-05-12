import signal
import threading
import uvicorn
import torch
from .parser import parser
from .diogen import diogen
from .api import app

int_signal = threading.Event()

def handle_sigint(_signum, _stack_frame):
    int_signal.set()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sigint)

    args = parser.parse_args()
    if args.use_accelerator and torch.accelerator.is_available():
        DEVICE = torch.accelerator.current_accelerator()
        diogen.to(DEVICE)

    uvicorn.run("src.server.main:app", host=args.host, port=args.port, workers=args.workers)