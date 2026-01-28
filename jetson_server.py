import socket

HOST = "0.0.0.0"
PORT = 7002

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:        
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(5)
        print(f"Jetson socket server listening on {HOST}:{PORT}")       

        while True:
            conn, addr = s.accept()
            with conn:
                data = conn.recv(1024)
                if not data:
                    continue
                msg = data.decode("utf-8", errors="ignore").strip()     
                print("Reçu:", msg)

                # TODO: déclencher action ici
                # ex: if msg == "left": ...

                conn.sendall(b"OK\n")

if __name__ == "__main__":
    main()





