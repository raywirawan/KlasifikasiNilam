from launch_backend import app

if __name__ == "__main__":
    print("Starting up (Gunicorn)...")
    app.run()