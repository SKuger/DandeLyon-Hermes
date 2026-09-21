FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# The corpus is part of the image: it is indexed at boot, and a missing
# directory turns every knowledge question into "I do not know".
COPY knowledge ./knowledge

# Not root: a public webhook is the part of the system most likely to be
# attacked, and it has no reason to own the filesystem.
RUN useradd --create-home bot
USER bot

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
