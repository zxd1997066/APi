python -m unittest api_test
python -m unittest c10d_rendezvous_backend_test
python -m unittest dynamic_rendezvous_test
python -m unittest etcd_rendezvous_backend_test
python -m unittest etcd_rendezvous_test
python -m unittest etcd_server_test
python -m unittest out_of_tree_rendezvous_test
python -m unittest rendezvous_backend_test
python -m unittest static_rendezvous_test
python -m unittest utils_test
cd ../agent/server/test
python -m unittest api_test