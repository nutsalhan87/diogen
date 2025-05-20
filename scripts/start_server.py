import os
import uvicorn
import torch

from server.parser import parser

if __name__ == "__main__":
    args = parser.parse_args()

    if args.use_accelerator and torch.accelerator.is_available():
        os.environ["DEVICE"] = torch.accelerator.current_accelerator().type
    else:
        os.environ["DEVICE"] = "cpu"

    uvicorn.run("server.api:app", host=args.host, port=args.port, workers=args.workers)
