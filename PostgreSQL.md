The error you're seeing (`could not translate host name "localhost" to address: nodename nor servname provided, or not known`) typically occurs when Django (or any other application) is unable to resolve the hostname `localhost` to the corresponding IP address (usually `127.0.0.1` for local connections).

### Here’s how to troubleshoot and resolve the issue:

### 1. **Check PostgreSQL Service Status**
Make sure that PostgreSQL is running. You can check its status and restart the service if necessary.

For macOS (if installed via Homebrew), run:

Must connect the CISCO VPN before running the below command

```bash
brew services list
```

If PostgreSQL is not running, start it with:

```bash
brew services start postgresql
```

If it's already running, try restarting it:

```bash
brew services restart postgresql
```

### 2. **Check PostgreSQL Host Configuration**

It’s possible that the PostgreSQL server is not properly configured to accept connections on `localhost`. To verify this, you need to check the `pg_hba.conf` file, which controls client authentication for PostgreSQL.

1. Locate your `pg_hba.conf` file. It is typically found at `/usr/local/var/postgres/pg_hba.conf` if you installed PostgreSQL using Homebrew.
   
2. Ensure that the file allows connections from `localhost`. The following line should be present:

   ```
   host    all             all             127.0.0.1/32            trust
   ```

   If this line is missing or incorrect, add it, save the file, and restart PostgreSQL:

   ```bash
   brew services restart postgresql
   ```

### 3. **Test Connection Using `psql`**

Try connecting to PostgreSQL directly using `psql`:

```bash
psql -h localhost -U Tayyab10 -d recommendations  # Replace with your actual database name
```

This will test if PostgreSQL is accessible via `localhost` using the correct user.

### 4. **Verify DNS Resolution for `localhost`**

If the PostgreSQL service is running correctly, and `localhost` is still not resolving, check the DNS resolution.

1. **Check `/etc/hosts` file**: Ensure that the `localhost` entry exists and points to `127.0.0.1`. Open the file and verify:

   ```bash
   cat /etc/hosts
   ```

   You should see a line like this:

   ```
   127.0.0.1   localhost
   ```

2. **Test hostname resolution**: Run this command to ensure `localhost` resolves to `127.0.0.1`:

   ```bash
   ping localhost
   ```

   You should get responses from `127.0.0.1`. If it doesn't work, the issue might be with your network configuration or `/etc/hosts`.

### 5. **Check the `DATABASES` Settings in Django**

Ensure that your `DATABASES` setting in `settings.py` is configured correctly. It should look like this:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'recommendations',  # Or the name of your database
        'USER': '......',  # Or the PostgreSQL user
        'PASSWORD': '....',  # The password for your PostgreSQL user
        'HOST': 'localhost',  # Should be 'localhost' or '127.0.0.1'
        'PORT': '5432',  # Default PostgreSQL port
    }
}
```

### 6. **Re-run Django Server**

Once the above configurations are verified and fixed, try running the Django development server again:

```bash
python manage.py runserver
```
