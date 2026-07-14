if [[ `hostname` == $1 ]]; then
    ray start --head --dashboard-host 0.0.0.0 --ray-client-server-port 33333
else
    ray start --address=$2:6379
fi
sleep 99999999999999