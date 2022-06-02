FROM python:3.7

COPY alibiexplainer alibiexplainer
COPY kserve kserve
COPY third_party third_party

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e ./kserve
RUN pip install --no-cache-dir -e ./alibiexplainer

RUN useradd kserve -m -u 1000 -d /home/kserve
USER 1000
ENTRYPOINT ["python", "-m", "alibiexplainer"]

# Gitlab:
# sudo -s
# docker login registry.gitlab.com
# docker build -t registry.gitlab.com/te4580/obligation-overwatch-test/kserve-alibi-explainer:test4 -f alibiexplainer.Dockerfile .
# docker push registry.gitlab.com/te4580/obligation-overwatch-test/kserve-alibi-explainer:test4
