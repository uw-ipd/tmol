#!/usr/bin/env bash
# Limit the read-only AtomWorks credential to the package build subprocess.
set +x
set -euo pipefail
: "${ATOMWORKS_DEPLOY_KEY:?Set the tmol Actions secret ATOMWORKS_DEPLOY_KEY}"
umask 077
auth_dir=$(mktemp -d)
trap 'rm -rf "$auth_dir"' EXIT
printf '%s\n' "$ATOMWORKS_DEPLOY_KEY" > "$auth_dir/key"
unset ATOMWORKS_DEPLOY_KEY APPTAINERENV_ATOMWORKS_DEPLOY_KEY
# GitHub's published Ed25519 host key (https://api.github.com/meta).
printf '%s\n' 'github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl' > "$auth_dir/known_hosts"
export GIT_SSH_COMMAND="ssh -F /dev/null -i \"$auth_dir/key\" -o IdentityAgent=none -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=\"$auth_dir/known_hosts\""
"$@"
