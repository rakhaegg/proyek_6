# Gunakan image Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Instal Poetry
RUN pip install poetry

# Salin file Poetry
COPY pyproject.toml poetry.lock /app/

# Instal dependensi menggunakan Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Salin semua file proyek
COPY . /app

# Expose port Flask
EXPOSE 5000

# Jalankan aplikasi Flask dengan Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "wsgi:app"]
