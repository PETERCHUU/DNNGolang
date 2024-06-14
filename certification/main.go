package main

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"net"
	"os"
	"time"
)

func main() {
	// Generate a CA private key
	caPrivateKey, _ := rsa.GenerateKey(rand.Reader, 2048)

	// Create a self-signed CA certificate
	caTemplate := x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			Organization: []string{"My CA"},
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(365 * 24 * time.Hour),
		KeyUsage:              x509.KeyUsageCertSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
	}
	caCertDER, _ := x509.CreateCertificate(rand.Reader, &caTemplate, &caTemplate, &caPrivateKey.PublicKey, caPrivateKey)
	caCertFile, _ := os.Create("ca.crt")
	pem.Encode(caCertFile, &pem.Block{Type: "CERTIFICATE", Bytes: caCertDER})
	caCertFile.Close()

	// Generate an IP SAN certificate
	ipSAN := "127.0.0.1"
	ipSANTemplate := x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject: pkix.Name{
			Organization: []string{"My Server"},
		},
		NotBefore:   time.Now(),
		NotAfter:    time.Now().Add(365 * 24 * time.Hour),
		KeyUsage:    x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		IPAddresses: []net.IP{net.ParseIP(ipSAN)},
		DNSNames:    []string{"localhost"},
	}
	ipSANCertDER, _ := x509.CreateCertificate(rand.Reader, &ipSANTemplate, &caTemplate, &caPrivateKey.PublicKey, caPrivateKey)
	ipSANCertFile, _ := os.Create("server.crt")
	pem.Encode(ipSANCertFile, &pem.Block{Type: "CERTIFICATE", Bytes: ipSANCertDER})
	ipSANCertFile.Close()
}
