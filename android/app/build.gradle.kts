import java.io.FileInputStream
import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// Version comes from the Python package, passed in by CI:
//   gradle assembleRelease -PmorphVersionName=0.1.3 -PmorphVersionCode=103
// so the APK and the agent that produced it always carry the same number.
val morphVersionName = (project.findProperty("morphVersionName") as String?) ?: "0.0.0"
val morphVersionCode = ((project.findProperty("morphVersionCode") as String?) ?: "1").toInt()

// Signing is optional. Without a keystore the release build is signed with the
// debug key, which still installs from a phone browser — it just cannot be
// shipped through Play. That keeps the loop able to cut a usable release with
// no secrets configured at all.
val keystoreFile = rootProject.file("keystore.jks")
val keystoreProps = rootProject.file("keystore.properties").let { file ->
    Properties().apply { if (file.exists()) load(FileInputStream(file)) }
}
val hasKeystore = keystoreFile.exists() && keystoreProps.getProperty("storePassword") != null

android {
    namespace = "dev.morph.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "dev.morph.app"
        minSdk = 24
        targetSdk = 34
        versionCode = morphVersionCode
        versionName = morphVersionName
    }

    signingConfigs {
        if (hasKeystore) {
            create("release") {
                storeFile = keystoreFile
                storePassword = keystoreProps.getProperty("storePassword")
                keyAlias = keystoreProps.getProperty("keyAlias")
                keyPassword = keystoreProps.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = if (hasKeystore) {
                signingConfigs.getByName("release")
            } else {
                signingConfigs.getByName("debug")
            }
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.2")
}
