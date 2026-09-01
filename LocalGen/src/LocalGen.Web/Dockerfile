# LocalGen web chat UI. Built from the repository root:
#   docker build -f src/LocalGen.Web/Dockerfile -t localgen-web .

FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

COPY Directory.Build.props Directory.Packages.props global.json ./
COPY src/LocalGen.Core/*.csproj  src/LocalGen.Core/
COPY src/LocalGen.Sdk/*.csproj   src/LocalGen.Sdk/
COPY src/LocalGen.Web/*.csproj   src/LocalGen.Web/

RUN dotnet restore src/LocalGen.Web/LocalGen.Web.csproj

COPY src/LocalGen.Core/ src/LocalGen.Core/
COPY src/LocalGen.Sdk/  src/LocalGen.Sdk/
COPY src/LocalGen.Web/  src/LocalGen.Web/

RUN dotnet publish src/LocalGen.Web/LocalGen.Web.csproj -c Release -o /app --no-restore

FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS runtime

RUN useradd --create-home --shell /usr/sbin/nologin localgen
WORKDIR /app
COPY --from=build /app .

USER localgen
EXPOSE 8080
ENV ASPNETCORE_HTTP_PORTS=8080

ENTRYPOINT ["dotnet", "LocalGen.Web.dll"]
