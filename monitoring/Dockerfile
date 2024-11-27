# Gunakan base image Prometheus resmi
FROM prom/prometheus:latest

# Salin konfigurasi Prometheus ke direktori kerja di container
COPY prometheus.yml /etc/prometheus/prometheus.yml

# Expose port 9090 untuk Prometheus
EXPOSE 9090

# Jalankan Prometheus
CMD ["--config.file=/etc/prometheus/prometheus.yml"]
