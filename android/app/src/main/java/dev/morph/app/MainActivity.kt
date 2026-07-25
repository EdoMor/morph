package dev.morph.app

import android.annotation.SuppressLint
import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.text.InputType
import android.view.View
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity

/**
 * Morph's phone client.
 *
 * Morph is self-hosted, so this app deliberately ships no server of its own: it
 * points at whatever machine is running `morph serve` — a laptop on the same
 * Wi-Fi, a Raspberry Pi, a forwarded Codespaces port, or Termux on this very
 * phone. The agent, the model and the conversation history stay on hardware the
 * user controls; this is the window onto it.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    private lateinit var progress: ProgressBar
    private lateinit var errorPanel: LinearLayout
    private lateinit var errorText: TextView

    private val prefs by lazy { getSharedPreferences(PREFS, Context.MODE_PRIVATE) }

    companion object {
        const val PREFS = "morph"
        const val KEY_SERVER = "server_url"
        const val DEFAULT_SERVER = "http://192.168.1.10:8787"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        webView = findViewById(R.id.webview)
        progress = findViewById(R.id.progress)
        errorPanel = findViewById(R.id.error_panel)
        errorText = findViewById(R.id.error_text)

        findViewById<Button>(R.id.retry_button).setOnClickListener { load() }
        findViewById<Button>(R.id.settings_button).setOnClickListener { promptForServer() }

        configureWebView()

        // Back navigates the web app before it leaves the activity, which is what
        // a chat UI with a drawer needs in order to feel native.
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (webView.canGoBack()) webView.goBack() else finish()
            }
        })

        if (serverUrl() == null) promptForServer() else load()
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun configureWebView() {
        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true // the client keeps its session id here
            databaseEnabled = true
            useWideViewPort = true
            loadWithOverviewMode = true
            cacheMode = WebSettings.LOAD_DEFAULT
            mediaPlaybackRequiresUserGesture = false
            mixedContentMode = WebSettings.MIXED_CONTENT_COMPATIBILITY_MODE
        }
        webView.setBackgroundColor(0xFF0B0D10.toInt())

        webView.webChromeClient = object : WebChromeClient() {
            override fun onProgressChanged(view: WebView?, newProgress: Int) {
                progress.progress = newProgress
                progress.visibility = if (newProgress in 1..99) View.VISIBLE else View.GONE
            }
        }

        webView.webViewClient = object : WebViewClient() {
            override fun onPageStarted(view: WebView?, url: String?, favicon: Bitmap?) {
                errorPanel.visibility = View.GONE
            }

            override fun shouldOverrideUrlLoading(
                view: WebView?,
                request: WebResourceRequest?
            ): Boolean {
                // Keep the Morph origin inside the app; hand anything else to the browser.
                val target = request?.url?.toString() ?: return false
                val server = serverUrl() ?: return false
                if (target.startsWith(server)) return false
                return try {
                    startActivity(Intent(Intent.ACTION_VIEW, request.url))
                    true
                } catch (_: Exception) {
                    false
                }
            }

            override fun onReceivedError(
                view: WebView?,
                request: WebResourceRequest?,
                error: WebResourceError?
            ) {
                if (request?.isForMainFrame != true) return
                showError(
                    "Cannot reach ${serverUrl()}\n\n" +
                        "Start it with `morph serve --host 0.0.0.0` and check that this " +
                        "phone is on the same network."
                )
            }
        }
    }

    private fun serverUrl(): String? = prefs.getString(KEY_SERVER, null)

    private fun load() {
        val url = serverUrl()
        if (url == null) {
            promptForServer()
            return
        }
        errorPanel.visibility = View.GONE
        webView.visibility = View.VISIBLE
        webView.loadUrl(url)
    }

    private fun showError(message: String) {
        errorText.text = message
        errorPanel.visibility = View.VISIBLE
        webView.visibility = View.GONE
        progress.visibility = View.GONE
    }

    private fun promptForServer() {
        val input = EditText(this).apply {
            inputType = InputType.TYPE_TEXT_VARIATION_URI
            setText(serverUrl() ?: DEFAULT_SERVER)
            setSelection(text.length)
        }
        val padding = (24 * resources.displayMetrics.density).toInt()
        val wrapper = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(padding, padding / 2, padding, 0)
            addView(input)
        }

        AlertDialog.Builder(this)
            .setTitle(R.string.server_title)
            .setMessage(R.string.server_message)
            .setView(wrapper)
            .setPositiveButton(R.string.connect) { _, _ ->
                var url = input.text.toString().trim().trimEnd('/')
                if (url.isEmpty()) return@setPositiveButton
                if (!url.startsWith("http://") && !url.startsWith("https://")) {
                    url = "http://$url"
                }
                prefs.edit().putString(KEY_SERVER, url).apply()
                load()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .setCancelable(serverUrl() != null)
            .show()
    }
}
