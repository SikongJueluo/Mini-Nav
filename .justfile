activate:
    micromamba activate ./.venv

update-venv:
    micromamba env export --no-builds | grep -v "prefix" > venv.yaml

download-test:
    python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
    python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data/
    python -m habitat_sim.utils.datasets_download --uids replica_cad_dataset --data-path data/
    python -m habitat_sim.utils.datasets_download --uids rearrange_dataset_v2 --data-path data/
    python -m habitat_sim.utils.datasets_download --uids hab_fetch --data-path data/
    python -m habitat_sim.utils.datasets_download --uids ycb --data-path data/
