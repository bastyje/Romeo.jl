FROM julia

WORKDIR /romeo

# Install dependencies
COPY Project.toml .
COPY test/bootstrap.jl test/bootstrap.jl
COPY src/Romeo.jl src/Romeo.jl
RUN ["julia", "--project=.", "/romeo/test/bootstrap.jl"]

COPY . .

# Run tests
CMD julia --project=. /romeo/test/runmnist.jl