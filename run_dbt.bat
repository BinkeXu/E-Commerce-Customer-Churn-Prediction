@echo off
set "GCP_PROJECT_ID=airy-web-484800-u5"
set "GOOGLE_APPLICATION_CREDENTIALS=%~dp0secrets\google-key.json"

echo Setting up dbt environment...
echo Project ID: %GCP_PROJECT_ID%
echo Key File: %GOOGLE_APPLICATION_CREDENTIALS%

cd dbt
call dbt build --profiles-dir .
if errorlevel 1 (
    echo dbt build failed!
    exit /b %errorlevel%
)
echo dbt build completed successfully!
cd ..
