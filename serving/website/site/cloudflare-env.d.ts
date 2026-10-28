declare namespace Cloudflare {
  interface Env {
    CLIQUEFLOWMER_API_ORIGIN?: string;
    CLIQUEFLOWMER_API_KEY?: string;
    CLIQUEFLOWMER_PUBLIC_ORIGIN?: string;
    DB?: D1Database;
    BUCKET?: R2Bucket;
  }
}
