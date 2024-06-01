from metagen import CompletionResponse, Message, MetaGenKey, thrift_platform_factory
from manifold.clients.python import ManifoldClient

# Create an authenticated MetaGenPlatform object with your MetaGen Key:


metagen_key = "mg-api-e9f0d342327b"

metagen = thrift_platform_factory.create_for_current_unix_user_for_devserver_only(
    metagen_auth_credential=MetaGenKey(key=metagen_key)
)

# Now you can generate content with a prompt:
prompt = "Tell me something about riding bikes"
response = metagen.chat_completion(
    messages=Message.message_list().add_user_message(prompt).build(),
)

print(f"Response: {response.choices[0].text}")

def manifold_download(
    model_manifold_bucket: str,
    model_manifold_dir: str,
    model_filename: str,
    local_path: str,
    override: bool = False,
    max_parallel_files: int = 50,
    retries: int = 5,
) -> None:
    remote_path = os.path.join(model_manifold_dir, model_filename)
    exist = os.path.exists(local_path)
    if exist:
        if not override:
            log.info("{} exists, skip the download".format(local_path))
            return
        else:
            shutil.rmtree(local_path)
            log.info("Purge the local path {}".format(local_path))
    else:
        log.info("local_path {} does not exist".format(local_path))

    os.makedirs(local_path, exist_ok=True)
    client = ManifoldClient.get_client(bucket=model_manifold_bucket)
    log.info(f"Create the local path {local_path}")
    log.info(f"Download from {remote_path} to {local_path}")

    backoff_time = 2
    for _ in range(retries):
        try:
            client.getRecursive(
                manifold_path=remote_path,
                local_path=local_path,
                maxParallelFiles=max_parallel_files,
            )
            log.info("Download is complete")
            client.close()
            return
        except Exception as e:
            print(e)
            log.warning("Quota Limit Exceeded, retrying...")

    log.error("Failed to download after multiple retries")
    client.close()