upload:
    rsync -avLh --progress --stats --itemize-changes \
    --exclude='.jj/' \
    --exclude='.git/' \
    --exclude='.devenv/' \
    --exclude='.direnv/' \
    --exclude='deps/' \
    --exclude='outputs/' \
    --exclude='data/versioned_data/' \
    . ial-jumper-ial-pangyg:/home/ial-pangyg/docker-workspace/projects/mini-nav/

sync-pkgs:
    export UV_PROJECT_ENVIRONMENT="/workspace/envs/mini-nav/" && uv sync --inexact

sync-data:
    python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
    python -m habitat_sim.utils.datasets_download --uids habitat_example_objects --data-path data/
