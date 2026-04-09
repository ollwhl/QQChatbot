#!/bin/bash
screen -dmS msg-server bash -c "python ./msg_server.py"
screen -dmS chatbot bash -c "python ./server.py"